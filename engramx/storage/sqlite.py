from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


class SQLiteDriver(BaseDriver):
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                scope TEXT NOT NULL,
                agent_id TEXT,
                user_id TEXT,
                session_id TEXT,
                content TEXT NOT NULL,
                payload TEXT,
                timestamp_created TEXT NOT NULL,
                timestamp_updated TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                source TEXT NOT NULL,
                provenance TEXT NOT NULL,
                derived_from TEXT NOT NULL,
                importance_score REAL NOT NULL,
                access_count INTEGER NOT NULL,
                last_accessed TEXT,
                decay_factor REAL NOT NULL,
                sensitivity_level TEXT NOT NULL,
                retention_policy TEXT,
                access_roles TEXT NOT NULL,
                embedding TEXT,
                embedding_model TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                actor TEXT NOT NULL,
                details TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _record_to_row(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "type": record.type,
            "scope": record.scope,
            "agent_id": record.agent_id,
            "user_id": record.user_id,
            "session_id": record.session_id,
            "content": record.content,
            "payload": _json_dump(record.payload),
            "timestamp_created": record.timestamp_created.isoformat(),
            "timestamp_updated": record.timestamp_updated.isoformat(),
            "valid_from": record.valid_from.isoformat() if record.valid_from else None,
            "valid_to": record.valid_to.isoformat() if record.valid_to else None,
            "source": record.source,
            "provenance": json.dumps(record.provenance),
            "derived_from": json.dumps(record.derived_from),
            "importance_score": record.importance_score,
            "access_count": record.access_count,
            "last_accessed": record.last_accessed.isoformat() if record.last_accessed else None,
            "decay_factor": record.decay_factor,
            "sensitivity_level": record.sensitivity_level,
            "retention_policy": record.retention_policy,
            "access_roles": json.dumps(record.access_roles),
            "embedding": _json_dump(record.embedding),
            "embedding_model": record.embedding_model,
        }

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord.from_dict(
            {
                "id": row["id"],
                "type": row["type"],
                "scope": row["scope"],
                "agent_id": row["agent_id"],
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "payload": json.loads(row["payload"]) if row["payload"] else None,
                "timestamp_created": row["timestamp_created"],
                "timestamp_updated": row["timestamp_updated"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "source": row["source"],
                "provenance": json.loads(row["provenance"]),
                "derived_from": json.loads(row["derived_from"]),
                "importance_score": row["importance_score"],
                "access_count": row["access_count"],
                "last_accessed": row["last_accessed"],
                "decay_factor": row["decay_factor"],
                "sensitivity_level": row["sensitivity_level"],
                "retention_policy": row["retention_policy"],
                "access_roles": json.loads(row["access_roles"]),
                "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                "embedding_model": row["embedding_model"],
            }
        )

    def _insert_event(self, event: MemoryTimelineEvent) -> None:
        self._conn.execute(
            "INSERT INTO events(operation, memory_id, timestamp, actor, details) VALUES (?, ?, ?, ?, ?)",
            (
                event.operation,
                event.memory_id,
                event.timestamp.isoformat(),
                event.actor,
                json.dumps(event.details),
            ),
        )

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        row = self._record_to_row(stored)
        columns = ", ".join(row.keys())
        placeholders = ", ".join(["?"] * len(row))
        assignments = ", ".join(f"{key}=excluded.{key}" for key in row if key != "id")
        self._conn.execute(
            f"INSERT INTO memories ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}",
            tuple(row.values()),
        )
        self._insert_event(
            MemoryTimelineEvent(
                operation="upsert",
                memory_id=stored.id,
                timestamp=stored.timestamp_updated,
                details={"record": stored.to_dict()},
            )
        )
        self._conn.commit()
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[RetrievedMemory]:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        records = [self._row_to_record(row) for row in rows]
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        records = [self._row_to_record(row) for row in rows]
        matched = [record for record in records if record_matches_filters(record, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        record = self._row_to_record(row).touch()
        record.access_count += 1
        record.last_accessed = utcnow()
        updated_row = self._record_to_row(record)
        assignments = ", ".join(f"{key}=?" for key in updated_row if key != "id")
        self._conn.execute(
            f"UPDATE memories SET {assignments} WHERE id = ?",
            tuple(value for key, value in updated_row.items() if key != "id") + (memory_id,),
        )
        self._insert_event(
            MemoryTimelineEvent(
                operation="get",
                memory_id=memory_id,
                timestamp=record.last_accessed,
                details={"record": record.to_dict()},
            )
        )
        self._conn.commit()
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        payload = self._row_to_record(row).to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)
        updated_row = self._record_to_row(updated)
        assignments = ", ".join(f"{key}=?" for key in updated_row if key != "id")
        self._conn.execute(
            f"UPDATE memories SET {assignments} WHERE id = ?",
            tuple(value for key, value in updated_row.items() if key != "id") + (memory_id,),
        )
        self._insert_event(
            MemoryTimelineEvent(
                operation="update",
                memory_id=memory_id,
                timestamp=updated.timestamp_updated,
                details={"record": updated.to_dict()},
            )
        )
        self._conn.commit()
        return updated

    async def delete(self, filters: dict[str, Any]) -> int:
        records = await self.list()
        ids = [record.id for record in records if record_matches_filters(record, filters)]
        for memory_id in ids:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            snapshot = self._row_to_record(row) if row is not None else None
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._insert_event(
                MemoryTimelineEvent(
                    operation="delete",
                    memory_id=memory_id,
                    timestamp=utcnow(),
                    details={"record": snapshot.to_dict() if snapshot else None},
                )
            )
        self._conn.commit()
        return len(ids)

    async def timeline(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        from_dt: str | None = None,
        to_dt: str | None = None,
        types: list[str] | None = None,
    ) -> list[MemoryTimelineEvent]:
        start = parse_datetime(from_dt)
        end = parse_datetime(to_dt)
        rows = self._conn.execute("SELECT * FROM events ORDER BY timestamp ASC, id ASC").fetchall()
        events: list[MemoryTimelineEvent] = []
        for row in rows:
            event = MemoryTimelineEvent(
                operation=row["operation"],
                memory_id=row["memory_id"],
                timestamp=parse_datetime(row["timestamp"]) or utcnow(),
                actor=row["actor"],
                details=json.loads(row["details"]),
            )
            payload = event.details.get("record")
            record = MemoryRecord.from_dict(payload) if payload else None
            if user_id and (record is None or record.user_id != user_id):
                continue
            if agent_id and (record is None or record.agent_id != agent_id):
                continue
            if session_id and (record is None or record.session_id != session_id):
                continue
            if types and (record is None or record.type not in types):
                continue
            if start and event.timestamp < start:
                continue
            if end and event.timestamp > end:
                continue
            events.append(event)
        return events

    async def explain(self, query: str, filters: dict[str, Any], top_k: int = 5) -> MemoryTrace:
        rows = self._conn.execute("SELECT * FROM memories").fetchall()
        records = [self._row_to_record(row) for row in rows]
        scored = score_records(records, query, filters)
        retrieved = scored[:top_k]
        return MemoryTrace(
            query=query,
            filters=filters,
            retrieved=[
                RetrievedMemory(
                    id=item.record.id,
                    score=item.score,
                    decay_adjusted=item.score * item.decay_component,
                    matched_terms=item.matched_terms,
                    policy_filters=item.policy_filters,
                )
                for item in retrieved
            ],
            candidates=scored,
        )
