"""Metrics schema and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    interrupt_count: int = Field(default=0, ge=0)
    approval_required: bool = False
    approval_observed: bool = False
    latency_ms: int = Field(default=0, ge=0)
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    avg_nodes_visited: float = Field(ge=0)
    total_retries: int = Field(ge=0)
    total_interrupts: int = Field(ge=0)
    resume_success: bool = False
    scenario_metrics: list[ScenarioMetric]

    @model_validator(mode="after")
    def totals_match_scenarios(self) -> MetricsReport:
        """Reject internally inconsistent reports before they reach grading."""
        if self.total_scenarios != len(self.scenario_metrics):
            raise ValueError("total_scenarios must match scenario_metrics length")
        return self


def metric_from_state(
    state: dict[str, Any],
    expected_route: str,
    approval_required: bool,
    latency_ms: int | None = None,
) -> ScenarioMetric:
    events = state.get("events", []) or []
    errors = state.get("errors", []) or []
    actual_route = state.get("route")
    approval = state.get("approval")
    nodes = [event.get("node", "unknown") for event in events]
    retry_count = sum(1 for node in nodes if node == "retry")
    interrupt_count = sum(1 for node in nodes if node == "approval")
    has_output = bool(state.get("final_answer") or state.get("pending_question"))
    success = actual_route == expected_route and has_output
    approval_observed = approval is not None or "approval" in nodes
    if approval_required:
        success = success and approval_observed
    measured_latency = (
        latency_ms
        if latency_ms is not None
        else sum(int(event.get("latency_ms", 0) or 0) for event in events)
    )
    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(nodes),
        retry_count=retry_count,
        interrupt_count=interrupt_count,
        approval_required=approval_required,
        approval_observed=approval_observed,
        latency_ms=measured_latency,
        errors=list(errors),
    )


def summarize_metrics(
    items: list[ScenarioMetric],
    *,
    resume_success: bool = False,
) -> MetricsReport:
    if not items:
        raise ValueError("No scenario metrics to summarize")
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        resume_success=resume_success,
        scenario_metrics=items,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
