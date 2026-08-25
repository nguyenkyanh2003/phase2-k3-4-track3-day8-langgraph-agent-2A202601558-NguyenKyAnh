"""Markdown report generation from validated workflow metrics."""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def _cell(value: object) -> str:
    """Escape values before placing them in a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report(metrics: MetricsReport) -> str:
    """Render a submission-ready lab report with evidence and analysis."""
    success_count = sum(metric.success for metric in metrics.scenario_metrics)
    summary_rows = [
        ("Total scenarios", metrics.total_scenarios),
        ("Successful scenarios", success_count),
        ("Success rate", f"{metrics.success_rate:.2%}"),
        ("Average nodes visited", f"{metrics.avg_nodes_visited:.2f}"),
        ("Total retries", metrics.total_retries),
        ("Approval/HITL events", metrics.total_interrupts),
        ("State-history recovery demonstrated", metrics.resume_success),
    ]
    summary_table = "\n".join(f"| {_cell(name)} | {_cell(value)} |" for name, value in summary_rows)

    scenario_rows: list[str] = []
    for metric in metrics.scenario_metrics:
        scenario_rows.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    metric.scenario_id,
                    metric.expected_route,
                    metric.actual_route or "",
                    "yes" if metric.success else "no",
                    metric.nodes_visited,
                    metric.retry_count,
                    metric.interrupt_count,
                    metric.latency_ms,
                )
            )
            + " |"
        )

    return f"""# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Student

- Name: Nguyen Ky Anh
- Project: Production-style support-ticket agent
- Runtime: Python 3.11, LangGraph, LangChain chat models, Pydantic

## 2. Architecture

The workflow separates probabilistic language understanding from deterministic safety and
routing. `classify` uses an LLM with a Pydantic structured-output schema. Routing functions
then select explicit graph edges. Tool results are judged by an LLM-as-judge, with an
additional deterministic rule that an explicit `ERROR` can never be accepted as success.
Every route terminates through the shared `finalize` audit node.

```mermaid
flowchart TD
    START --> intake --> classify
    classify -->|simple| answer
    classify -->|tool| tool
    classify -->|missing_info| clarify
    classify -->|risky| risky_action --> approval
    classify -->|error| retry
    approval -->|approved| tool
    approval -->|rejected| clarify
    tool --> evaluate
    evaluate -->|success| answer
    evaluate -->|needs_retry| retry
    retry -->|attempt < max| tool
    retry -->|attempt >= max| dead_letter
    answer --> finalize
    clarify --> finalize
    dead_letter --> finalize --> END
```

## 3. State schema and reducers

| Field | Update behavior | Purpose |
|---|---|---|
| `query`, `route`, `risk_level` | overwrite | Current normalized request and routing decision |
| `attempt`, `max_attempts`, `evaluation_result` | overwrite | Bounded retry-loop control |
| `pending_question`, `proposed_action`, `approval` | overwrite | HITL safety state |
| `final_answer`, `classification_reason` | overwrite | Final output and explainability |
| `messages`, `tool_results`, `errors`, `events` | append reducer | Serializable audit history |

## 4. Metrics summary

| Metric | Value |
|---|---:|
{summary_table}

## 5. Scenario results

| Scenario | Expected | Actual | Success | Nodes | Retries | HITL | Latency ms |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

## 6. Failure analysis

1. **Transient tool failure:** explicit error results are classified as `needs_retry`.
   `attempt` is incremented in one node and checked in a separate routing function, so the
   loop cannot run forever. Exhaustion routes to `dead_letter` and manual escalation.
2. **Risky action without approval:** side-effecting requests are prepared but not executed
   before the approval node. The tool also has a defense-in-depth check that blocks a risky
   action when approval is absent, even if graph wiring is accidentally changed.
3. **Provider outage or malformed output:** LLM calls use structured Pydantic schemas,
   request timeouts, bounded provider retries, and audited conservative fallbacks.
4. **Incomplete request:** vague input terminates with a targeted clarification question
   rather than hallucinating missing account or order details.

## 7. Persistence and recovery evidence

The lab configuration uses SQLite with WAL mode. Every graph invocation receives a unique
`thread_id`. Automated tests execute a workflow, close the SQLite connection, create a new
checkpointer, and verify that the final state and complete state history are still available.
State-history replay observed during scenario execution: **{metrics.resume_success}**.

## 8. Extensions completed

- Real optional HITL with `interrupt()` when `LANGGRAPH_INTERRUPT=true`.
- SQLite crash/restart recovery and optional PostgreSQL adapter.
- LLM-as-judge structured evaluation.
- Mermaid graph diagram embedded in this report and exportable from the CLI.
- Per-node audit events, provider latency, retry/error, and approval metrics.
- Provider-independent LLM factory for Gemini, OpenAI, and Anthropic.

## 9. Improvement plan

The next production step would replace the mock tool with authenticated, idempotent service
clients and durable action IDs. Then add distributed tracing, prompt/version evaluation,
role-based approval, PII redaction, rate limiting, and adversarial hidden-scenario tests.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a UTF-8 Markdown file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
