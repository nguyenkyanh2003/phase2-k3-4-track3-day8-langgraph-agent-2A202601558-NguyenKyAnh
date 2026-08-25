"""Persistence adapters for LangGraph checkpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver


def _sqlite_database_path(database_url: str | None) -> str:
    """Normalize a local path or ``sqlite:///`` URL for ``sqlite3.connect``."""
    value = database_url or "checkpoints.db"
    if value == ":memory:":
        return value
    if value.startswith("sqlite:///"):
        value = value.removeprefix("sqlite:///")
    elif value.startswith("sqlite://"):
        raise ValueError("SQLite URL must use sqlite:///path/to/database.db")

    path = Path(value).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_checkpointer(
    kind: str = "memory",
    database_url: str | None = None,
) -> BaseCheckpointSaver | None:
    """Build a sync checkpointer for memory, SQLite, or PostgreSQL.

    SQLite uses WAL mode and a busy timeout for safer multi-step CLI runs. The
    caller owns the saver and may close ``saver.conn`` when its lifecycle ends.
    """
    normalized_kind = kind.casefold()
    if normalized_kind == "none":
        return None
    if normalized_kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if normalized_kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        database_path = _sqlite_database_path(database_url)
        connection = sqlite3.connect(database_path, check_same_thread=False)
        connection.execute("PRAGMA busy_timeout=5000")
        if database_path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
        saver = SqliteSaver(conn=connection)
        saver.setup()
        return saver
    if normalized_kind == "postgres":
        if not database_url:
            raise ValueError("database_url is required for the Postgres checkpointer")
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
        except ImportError as exc:
            raise RuntimeError(
                "Install PostgreSQL support with: pip install -e '.[postgres]'"
            ) from exc

        connection = Connection.connect(database_url, autocommit=True, prepare_threshold=0)
        saver = PostgresSaver(connection)
        saver.setup()
        return saver
    raise ValueError(f"Unknown checkpointer kind: {kind}")


def close_checkpointer(checkpointer: BaseCheckpointSaver | None) -> None:
    """Close an owned database connection when a saver exposes one."""
    connection = getattr(checkpointer, "conn", None)
    close = getattr(connection, "close", None)
    if callable(close):
        close()
