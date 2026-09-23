from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping


class SQLiteExperimentStore:
    """Append-only audit events plus versioned engine snapshots."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_index INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def append(self, event: str, payload: Mapping[str, Any]) -> None:
        serialized = json.dumps(dict(payload), sort_keys=True)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events(event, payload) VALUES (?, ?)",
                (event, serialized),
            )

    def save_snapshot(self, round_index: int, payload: Mapping[str, Any]) -> int:
        serialized = json.dumps(dict(payload), sort_keys=True)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO snapshots(round_index, payload) VALUES (?, ?)",
                (round_index, serialized),
            )
            return int(cursor.lastrowid)

    def latest_snapshot(self) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
        return json.loads(row[0]) if row else None

    def events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, event, payload, created_at "
                "FROM events ORDER BY sequence"
            ).fetchall()
        return [
            {
                "sequence": row[0],
                "event": row[1],
                "payload": json.loads(row[2]),
                "created_at": row[3],
            }
            for row in rows
        ]
