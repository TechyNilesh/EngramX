from __future__ import annotations

import importlib.util
import json
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("neo4j",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _record_to_props(record: MemoryRecord) -> dict[str, Any]:
    """Convert a MemoryRecord to a flat dict of Neo4j-compatible properties."""
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


def _node_to_record(node: dict[str, Any]) -> MemoryRecord:
    """Convert a Neo4j node property dict back to a MemoryRecord."""
    return MemoryRecord.from_dict(
        {
            "id": node["id"],
            "type": node["type"],
            "scope": node["scope"],
            "agent_id": node.get("agent_id"),
            "user_id": node.get("user_id"),
            "session_id": node.get("session_id"),
            "content": node["content"],
            "payload": json.loads(node["payload"]) if node.get("payload") else None,
            "timestamp_created": node["timestamp_created"],
            "timestamp_updated": node["timestamp_updated"],
            "valid_from": node.get("valid_from"),
            "valid_to": node.get("valid_to"),
            "source": node["source"],
            "provenance": json.loads(node["provenance"]),
            "derived_from": json.loads(node["derived_from"]),
            "importance_score": node["importance_score"],
            "access_count": node["access_count"],
            "last_accessed": node.get("last_accessed"),
            "decay_factor": node["decay_factor"],
            "sensitivity_level": node["sensitivity_level"],
            "retention_policy": node.get("retention_policy"),
            "access_roles": json.loads(node["access_roles"]),
            "embedding": json.loads(node["embedding"]) if node.get("embedding") else None,
            "embedding_model": node.get("embedding_model"),
        }
    )


class Neo4jDriver(BaseDriver):
    """Neo4j-backed memory storage driver.

    Uses the official ``neo4j`` async Python driver. Memories are stored as
    ``:Memory`` nodes and timeline events as ``:MemoryEvent`` nodes.
    ``DERIVED_FROM`` relationships link derived memories to their parents.
    """

    def __init__(
        self,
        uri: str,
        auth: tuple[str, str] = ("neo4j", "neo4j"),
        database: str = "neo4j",
    ) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "Neo4jDriver requires the optional `neo4j` dependency. "
                "Install the Neo4j extra before constructing this driver."
            )
        self.uri = uri
        self.auth = auth
        self.database = database
        self._driver: Any = None  # neo4j.AsyncDriver, lazily created
        self._initialised = False

    async def _ensure_driver(self) -> None:
        """Lazily create the async Neo4j driver and apply schema constraints."""
        if self._driver is not None and self._initialised:
            return

        from neo4j import AsyncGraphDatabase  # type: ignore[import-untyped]

        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(self.uri, auth=self.auth)

        if not self._initialised:
            async with self._driver.session(database=self.database) as session:
                await session.run(
                    "CREATE CONSTRAINT memory_id_unique IF NOT EXISTS "
                    "FOR (m:Memory) REQUIRE m.id IS UNIQUE"
                )
                await session.run(
                    "CREATE CONSTRAINT event_id_unique IF NOT EXISTS "
                    "FOR (e:MemoryEvent) REQUIRE e.event_id IS UNIQUE"
                )
            self._initialised = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher query and return a list of record dicts."""
        await self._ensure_driver()
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, parameters or {})
            return [record.data() async for record in result]

    async def _write_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute a Cypher write query inside an explicit write transaction."""
        await self._ensure_driver()
        async with self._driver.session(database=self.database) as session:

            async def _work(tx: Any) -> list[dict[str, Any]]:
                result = await tx.run(query, parameters or {})
                return [record.data() async for record in result]

            return await session.execute_write(_work)

    async def _insert_event(self, event: MemoryTimelineEvent) -> None:
        """Persist a :MemoryEvent node."""
        import uuid as _uuid

        props = {
            "event_id": str(_uuid.uuid4()),
            "operation": event.operation,
            "memory_id": event.memory_id,
            "timestamp": event.timestamp.isoformat(),
            "actor": event.actor,
            "details": json.dumps(event.details),
        }
        await self._write_query(
            "CREATE (e:MemoryEvent $props)",
            {"props": props},
        )

    async def _manage_derived_from(self, record: MemoryRecord) -> None:
        """Ensure DERIVED_FROM edges exist between this memory and its parents."""
        if not record.derived_from:
            return
        # Remove stale edges first, then create correct ones
        await self._write_query(
            "MATCH (child:Memory {id: $id})-[r:DERIVED_FROM]->() DELETE r",
            {"id": record.id},
        )
        for parent_id in record.derived_from:
            await self._write_query(
                "MATCH (child:Memory {id: $child_id}), (parent:Memory {id: $parent_id}) "
                "MERGE (child)-[:DERIVED_FROM]->(parent)",
                {"child_id": record.id, "parent_id": parent_id},
            )

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    # ------------------------------------------------------------------
    # BaseDriver implementation
    # ------------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        props = _record_to_props(stored)
        await self._write_query(
            "MERGE (m:Memory {id: $props.id}) "
            "SET m = $props",
            {"props": props},
        )
        await self._manage_derived_from(stored)
        await self._insert_event(
            MemoryTimelineEvent(
                operation="upsert",
                memory_id=stored.id,
                timestamp=stored.timestamp_updated,
                details=self._snapshot(stored),
            )
        )
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[ScoredMemory]:
        rows = await self._run_query("MATCH (m:Memory) RETURN properties(m) AS props")
        records = [_node_to_record(row["props"]) for row in rows]
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        rows = await self._run_query("MATCH (m:Memory) RETURN properties(m) AS props")
        records = [_node_to_record(row["props"]) for row in rows]
        matched = [record for record in records if record_matches_filters(record, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        rows = await self._run_query(
            "MATCH (m:Memory {id: $id}) RETURN properties(m) AS props",
            {"id": memory_id},
        )
        if not rows:
            raise KeyError(memory_id)
        record = _node_to_record(rows[0]["props"]).touch()
        record.access_count += 1
        record.last_accessed = utcnow()
        # Persist the access update
        props = _record_to_props(record)
        await self._write_query(
            "MATCH (m:Memory {id: $id}) SET m = $props",
            {"id": memory_id, "props": props},
        )
        await self._insert_event(
            MemoryTimelineEvent(
                operation="get",
                memory_id=memory_id,
                timestamp=record.last_accessed,
                details=self._snapshot(record),
            )
        )
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        rows = await self._run_query(
            "MATCH (m:Memory {id: $id}) RETURN properties(m) AS props",
            {"id": memory_id},
        )
        if not rows:
            raise KeyError(memory_id)
        payload = _node_to_record(rows[0]["props"]).to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)
        props = _record_to_props(updated)
        await self._write_query(
            "MATCH (m:Memory {id: $id}) SET m = $props",
            {"id": memory_id, "props": props},
        )
        await self._manage_derived_from(updated)
        await self._insert_event(
            MemoryTimelineEvent(
                operation="update",
                memory_id=memory_id,
                timestamp=updated.timestamp_updated,
                details=self._snapshot(updated),
            )
        )
        return updated

    async def delete(self, filters: dict[str, Any]) -> int:
        all_records = await self.list()
        ids = [record.id for record in all_records if record_matches_filters(record, filters)]
        for memory_id in ids:
            # Fetch snapshot before deletion
            rows = await self._run_query(
                "MATCH (m:Memory {id: $id}) RETURN properties(m) AS props",
                {"id": memory_id},
            )
            snapshot = _node_to_record(rows[0]["props"]) if rows else None
            # Delete the node and all its relationships
            await self._write_query(
                "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                {"id": memory_id},
            )
            await self._insert_event(
                MemoryTimelineEvent(
                    operation="delete",
                    memory_id=memory_id,
                    timestamp=utcnow(),
                    details=self._snapshot(snapshot),
                )
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
        start = parse_datetime(from_dt)
        end = parse_datetime(to_dt)

        rows = await self._run_query(
            "MATCH (e:MemoryEvent) RETURN properties(e) AS props ORDER BY e.timestamp ASC"
        )

        events: list[MemoryTimelineEvent] = []
        for row in rows:
            p = row["props"]
            event = MemoryTimelineEvent(
                operation=p["operation"],
                memory_id=p["memory_id"],
                timestamp=parse_datetime(p["timestamp"]) or utcnow(),
                actor=p.get("actor", "system"),
                details=json.loads(p["details"]),
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
        rows = await self._run_query("MATCH (m:Memory) RETURN properties(m) AS props")
        records = [_node_to_record(row["props"]) for row in rows]
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
        """Shut down the underlying Neo4j async driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            self._initialised = False
