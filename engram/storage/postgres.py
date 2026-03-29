from __future__ import annotations

import importlib.util
import json
from typing import Any

from ..embedding import EMBEDDING_DIM, embed_record, embed_text
from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records


def _dependency_available() -> bool:
    return importlib.util.find_spec("asyncpg") is not None


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


class PostgresDriver(BaseDriver):
    """Async PostgreSQL driver using asyncpg with pgvector support."""

    def __init__(self, dsn: str, *, table_name: str = "engram_memories") -> None:
        if not _dependency_available():
            raise RuntimeError(
                "PostgresDriver requires the 'asyncpg' package. "
                "Install it with: pip install asyncpg"
            )
        self.dsn = dsn
        self.table_name = table_name
        self._pool: Any = None  # asyncpg.Pool

    async def connect(self) -> None:
        """Create the connection pool and initialize tables."""
        import asyncpg  # noqa: F811

        self._pool = await asyncpg.create_pool(dsn=self.dsn)
        await self._init_db()

    async def _ensure_pool(self) -> None:
        if self._pool is None:
            await self.connect()

    async def _init_db(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    agent_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    payload JSONB,
                    timestamp_created TEXT NOT NULL,
                    timestamp_updated TEXT NOT NULL,
                    valid_from TEXT,
                    valid_to TEXT,
                    source TEXT NOT NULL,
                    provenance JSONB NOT NULL DEFAULT '[]'::jsonb,
                    derived_from JSONB NOT NULL DEFAULT '[]'::jsonb,
                    importance_score DOUBLE PRECISION NOT NULL,
                    access_count INTEGER NOT NULL,
                    last_accessed TEXT,
                    decay_factor DOUBLE PRECISION NOT NULL,
                    sensitivity_level TEXT NOT NULL,
                    retention_policy TEXT,
                    access_roles JSONB NOT NULL DEFAULT '[]'::jsonb,
                    embedding vector({EMBEDDING_DIM}),
                    embedding_model TEXT
                )
                """
            )
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name}_events (
                    id BIGSERIAL PRIMARY KEY,
                    operation TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details JSONB NOT NULL DEFAULT '{{}}'::jsonb
                )
                """
            )

    def _record_to_row(self, record: MemoryRecord) -> dict[str, Any]:
        embedding_str: str | None = None
        if record.embedding is not None:
            embedding_str = "[" + ",".join(str(v) for v in record.embedding) + "]"
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
            "embedding": embedding_str,
            "embedding_model": record.embedding_model,
        }

    def _row_to_record(self, row: dict[str, Any] | Any) -> MemoryRecord:
        # asyncpg returns Record objects that support key access
        payload_val = row["payload"]
        if isinstance(payload_val, str):
            payload_val = json.loads(payload_val)

        provenance_val = row["provenance"]
        if isinstance(provenance_val, str):
            provenance_val = json.loads(provenance_val)

        derived_from_val = row["derived_from"]
        if isinstance(derived_from_val, str):
            derived_from_val = json.loads(derived_from_val)

        access_roles_val = row["access_roles"]
        if isinstance(access_roles_val, str):
            access_roles_val = json.loads(access_roles_val)

        # pgvector returns embedding as a string like '[0.1,0.2,...]' or as a list
        embedding_raw = row["embedding"]
        embedding: list[float] | None = None
        if embedding_raw is not None:
            if isinstance(embedding_raw, str):
                embedding = [float(v) for v in embedding_raw.strip("[]").split(",")]
            elif isinstance(embedding_raw, (list, tuple)):
                embedding = [float(v) for v in embedding_raw]
            else:
                # numpy array or pgvector type -- convert via list()
                embedding = [float(v) for v in list(embedding_raw)]

        return MemoryRecord.from_dict(
            {
                "id": row["id"],
                "type": row["type"],
                "scope": row["scope"],
                "agent_id": row["agent_id"],
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "content": row["content"],
                "payload": payload_val,
                "timestamp_created": row["timestamp_created"],
                "timestamp_updated": row["timestamp_updated"],
                "valid_from": row["valid_from"],
                "valid_to": row["valid_to"],
                "source": row["source"],
                "provenance": provenance_val,
                "derived_from": derived_from_val,
                "importance_score": row["importance_score"],
                "access_count": row["access_count"],
                "last_accessed": row["last_accessed"],
                "decay_factor": row["decay_factor"],
                "sensitivity_level": row["sensitivity_level"],
                "retention_policy": row["retention_policy"],
                "access_roles": access_roles_val,
                "embedding": embedding,
                "embedding_model": row["embedding_model"],
            }
        )

    async def _insert_event(self, conn: Any, event: MemoryTimelineEvent) -> None:
        await conn.execute(
            f"""
            INSERT INTO {self.table_name}_events (operation, memory_id, timestamp, actor, details)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            event.operation,
            event.memory_id,
            event.timestamp.isoformat(),
            event.actor,
            json.dumps(event.details),
        )

    async def upsert(self, record: MemoryRecord) -> str:
        await self._ensure_pool()
        stored = record.touch()
        row = self._record_to_row(stored)
        columns = list(row.keys())
        col_str = ", ".join(columns)
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        # Cast the embedding placeholder to vector type
        placeholders_list = [f"${i + 1}" for i in range(len(columns))]
        emb_idx = columns.index("embedding")
        placeholders_list[emb_idx] = f"${emb_idx + 1}::vector"
        # Cast JSONB columns
        for jsonb_col in ("payload", "provenance", "derived_from", "access_roles"):
            if jsonb_col in columns:
                idx = columns.index(jsonb_col)
                placeholders_list[idx] = f"${idx + 1}::jsonb"
        placeholders = ", ".join(placeholders_list)
        assignments = ", ".join(
            f"{col}=EXCLUDED.{col}" for col in columns if col != "id"
        )
        sql = (
            f"INSERT INTO {self.table_name} ({col_str}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}"
        )
        values = tuple(row.values())

        async with self._pool.acquire() as conn:
            await conn.execute(sql, *values)
            await self._insert_event(
                conn,
                MemoryTimelineEvent(
                    operation="upsert",
                    memory_id=stored.id,
                    timestamp=stored.timestamp_updated,
                    details={"record": stored.to_dict()},
                ),
            )
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[Any]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {self.table_name}")
        records = [self._row_to_record(row) for row in rows]
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {self.table_name}")
        records = [self._row_to_record(row) for row in rows]
        matched = [record for record in records if record_matches_filters(record, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} WHERE id = $1", memory_id
            )
            if row is None:
                raise KeyError(memory_id)
            record = self._row_to_record(row).touch()
            record.access_count += 1
            record.last_accessed = utcnow()
            updated_row = self._record_to_row(record)
            columns = list(updated_row.keys())
            set_parts = []
            values: list[Any] = []
            param_idx = 1
            for col in columns:
                if col == "id":
                    continue
                cast = ""
                if col == "embedding":
                    cast = "::vector"
                elif col in ("payload", "provenance", "derived_from", "access_roles"):
                    cast = "::jsonb"
                set_parts.append(f"{col}=${param_idx}{cast}")
                values.append(updated_row[col])
                param_idx += 1
            values.append(memory_id)
            await conn.execute(
                f"UPDATE {self.table_name} SET {', '.join(set_parts)} WHERE id = ${param_idx}",
                *values,
            )
            await self._insert_event(
                conn,
                MemoryTimelineEvent(
                    operation="get",
                    memory_id=memory_id,
                    timestamp=record.last_accessed,
                    details={"record": record.to_dict()},
                ),
            )
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} WHERE id = $1", memory_id
            )
            if row is None:
                raise KeyError(memory_id)
            payload = self._row_to_record(row).to_dict()
            payload.update(patch)
            payload["timestamp_updated"] = utcnow().isoformat()
            updated = MemoryRecord.from_dict(payload)
            updated_row = self._record_to_row(updated)
            columns = list(updated_row.keys())
            set_parts = []
            values: list[Any] = []
            param_idx = 1
            for col in columns:
                if col == "id":
                    continue
                cast = ""
                if col == "embedding":
                    cast = "::vector"
                elif col in ("payload", "provenance", "derived_from", "access_roles"):
                    cast = "::jsonb"
                set_parts.append(f"{col}=${param_idx}{cast}")
                values.append(updated_row[col])
                param_idx += 1
            values.append(memory_id)
            await conn.execute(
                f"UPDATE {self.table_name} SET {', '.join(set_parts)} WHERE id = ${param_idx}",
                *values,
            )
            await self._insert_event(
                conn,
                MemoryTimelineEvent(
                    operation="update",
                    memory_id=memory_id,
                    timestamp=updated.timestamp_updated,
                    details={"record": updated.to_dict()},
                ),
            )
        return updated

    async def delete(self, filters: dict[str, Any]) -> int:
        await self._ensure_pool()
        records = await self.list()
        ids = [record.id for record in records if record_matches_filters(record, filters)]
        if not ids:
            return 0
        async with self._pool.acquire() as conn:
            for memory_id in ids:
                row = await conn.fetchrow(
                    f"SELECT * FROM {self.table_name} WHERE id = $1", memory_id
                )
                snapshot = self._row_to_record(row) if row is not None else None
                await conn.execute(
                    f"DELETE FROM {self.table_name} WHERE id = $1", memory_id
                )
                await self._insert_event(
                    conn,
                    MemoryTimelineEvent(
                        operation="delete",
                        memory_id=memory_id,
                        timestamp=utcnow(),
                        details={"record": snapshot.to_dict() if snapshot else None},
                    ),
                )
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
        await self._ensure_pool()
        start = parse_datetime(from_dt)
        end = parse_datetime(to_dt)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.table_name}_events ORDER BY timestamp ASC, id ASC"
            )
        events: list[MemoryTimelineEvent] = []
        for row in rows:
            details_raw = row["details"]
            if isinstance(details_raw, str):
                details_raw = json.loads(details_raw)
            event = MemoryTimelineEvent(
                operation=row["operation"],
                memory_id=row["memory_id"],
                timestamp=parse_datetime(row["timestamp"]) or utcnow(),
                actor=row["actor"],
                details=details_raw,
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
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {self.table_name}")
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

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
