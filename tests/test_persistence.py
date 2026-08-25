"""SQLite checkpoint persistence and recovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

import langgraph_agent_lab.nodes as nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class OfflineLLM:
    """Force the nodes' documented deterministic resilience path."""

    def with_structured_output(self, _schema: object) -> OfflineLLM:
        return self

    def invoke(self, _prompt: str) -> object:
        raise RuntimeError("offline persistence test")


def test_memory_and_none_checkpointers() -> None:
    assert build_checkpointer("memory") is not None
    assert build_checkpointer("none") is None


def test_unknown_checkpointer_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown checkpointer"):
        build_checkpointer("unsupported")


def test_sqlite_state_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(nodes, "get_llm", lambda **_kwargs: OfflineLLM())
    database_path = tmp_path / "recovery.db"
    scenario = Scenario(
        id="sqlite-recovery",
        query="How do I reset my password?",
        expected_route=Route.SIMPLE,
    )
    initial = initial_state(scenario)
    config = {"configurable": {"thread_id": initial["thread_id"]}}

    first_saver = build_checkpointer("sqlite", str(database_path))
    assert isinstance(first_saver, SqliteSaver)
    first_graph = build_graph(first_saver)
    completed = first_graph.invoke(initial, config=config)
    history_before_close = list(first_graph.get_state_history(config))
    first_saver.conn.close()

    second_saver = build_checkpointer("sqlite", f"sqlite:///{database_path}")
    assert isinstance(second_saver, SqliteSaver)
    try:
        recovered_graph = build_graph(second_saver)
        recovered = recovered_graph.get_state(config)
        history_after_reopen = list(recovered_graph.get_state_history(config))

        assert recovered.values["final_answer"] == completed["final_answer"]
        assert recovered.values["events"][-1]["node"] == "finalize"
        assert len(history_before_close) >= 2
        assert len(history_after_reopen) == len(history_before_close)
    finally:
        second_saver.conn.close()
