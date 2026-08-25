"""Lab report rendering tests."""

from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report


def test_render_report_contains_required_evidence() -> None:
    metrics = MetricsReport(
        total_scenarios=1,
        success_rate=1.0,
        avg_nodes_visited=4.0,
        total_retries=0,
        total_interrupts=0,
        resume_success=True,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="S|01",
                success=True,
                expected_route="simple",
                actual_route="simple",
                nodes_visited=4,
                latency_ms=25,
            )
        ],
    )

    report = render_report(metrics)

    assert "```mermaid" in report
    assert "State schema and reducers" in report
    assert "Failure analysis" in report
    assert "Persistence and recovery evidence" in report
    assert "S\\|01" in report
    assert "100.00%" in report
