"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Annotated
from uuid import uuid4

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer, close_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    state_history_observed = False
    try:
        for scenario in scenarios:
            state = initial_state(scenario)
            state["thread_id"] = f"thread-{scenario.id}-{uuid4().hex[:12]}"
            run_config = {"configurable": {"thread_id": state["thread_id"]}}
            started_at = perf_counter()
            final_state = graph.invoke(state, config=run_config)
            latency_ms = max(0, round((perf_counter() - started_at) * 1000))
            if checkpointer is not None:
                state_history_observed = state_history_observed or len(
                    list(graph.get_state_history(run_config))
                ) >= 2
            metrics.append(
                metric_from_state(
                    final_state,
                    scenario.expected_route.value,
                    scenario.requires_approval,
                    latency_ms,
                )
            )
        report = summarize_metrics(metrics, resume_success=state_history_observed)
        write_metrics(report, output)
        if cfg.get("report_path"):
            write_report(report, cfg["report_path"])
    finally:
        close_checkpointer(checkpointer)
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("export-diagram")
def export_diagram(
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/graph.mmd"),
) -> None:
    """Export the compiled workflow as a Mermaid diagram."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
    typer.echo(f"Wrote Mermaid graph to {output}")


if __name__ == "__main__":
    app()
