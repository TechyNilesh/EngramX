from __future__ import annotations

import importlib.util
import json
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("redis",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


def _json_dump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


# Fields that hold JSON-serialised complex objects in the Redis hash.
_JSON_FIELDS = frozenset(
    {
        "payload",
        "provenance",
        "derived_from",
        "access_roles",
        "embedding",
    }
)

# Fields that are optional and may legitimately be ``None``.
_OPTIONAL_FIELDS = frozenset(
    {
        "agent_id",
        "user_id",
        "session_id",
        "payload",
        "valid_from",
        "valid_to",
        "last_accessed",
        "retention_policy",
        "embedding",
        "embedding_model",
    }
)


class RedisDriver(BaseDriver):
    """Async Redis-backed storage driver for Engram memory records.

    Each :class:`MemoryRecord` is stored as a Redis hash under the key
    ``{prefix}mem:{id}``.  A Redis set at ``{prefix}ids`` tracks all known
    memory IDs, and a Redis list at ``{prefix}events`` stores serialised
    timeline events in insertion order.
    """

    def __init__(self, url: str = "redis://localhost:6379", prefix: str = "engram:") -> None:
        if not _dependency_available():
            raise RuntimeError(
                "RedisDriver requires the optional `redis` dependency. "
                "Install the Redis extra before constructing this driver."
            )
        import redis.asyncio as aioredis

        self.url = url
        self.prefix = prefix
        self._redis: aioredis.Redis = aioredis.from_url(url, decode_responses=True)

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    def _mem_key(self, memory_id: str) -> str:
        return f"{self.prefix}mem:{memory_id}"

    @property
    def _ids_key(self) -> str:
        return f"{self.prefix}ids"

    @property
    def _events_key(self) -> str:
        return f"{self.prefix}events"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _record_to_hash(self, record: MemoryRecord) -> dict[str, str]:
        """Flatten a *MemoryRecord* into a ``str -> str`` mapping suitable for
        ``HSET``.  Complex / nullable fields are JSON-encoded; datetime fields
        are stored as ISO-8601 strings."""
        mapping: dict[str, str] = {
            "id": record.id,
            "type": record.type,
            "scope": record.scope,
            "content": record.content,
            "timestamp_created": record.timestamp_created.isoformat(),
            "timestamp_updated": record.timestamp_updated.isoformat(),
            "source": record.source,
            "provenance": json.dumps(record.provenance),
            "derived_from": json.dumps(record.derived_from),
            "importance_score": str(record.importance_score),
            "access_count": str(record.access_count),
            "decay_factor": str(record.decay_factor),
            "sensitivity_level": record.sensitivity_level,
            "access_roles": json.dumps(record.access_roles),
        }
        # Optional scalar fields
        if record.agent_id is not None:
            mapping["agent_id"] = record.agent_id
        if record.user_id is not None:
            mapping["user_id"] = record.user_id
        if record.session_id is not None:
            mapping["session_id"] = record.session_id
        if record.payload is not None:
            mapping["payload"] = json.dumps(record.payload)
        if record.valid_from is not None:
            mapping["valid_from"] = record.valid_from.isoformat()
        if record.valid_to is not None:
            mapping["valid_to"] = record.valid_to.isoformat()
        if record.last_accessed is not None:
            mapping["last_accessed"] = record.last_accessed.isoformat()
        if record.retention_policy is not None:
            mapping["retention_policy"] = record.retention_policy
        if record.embedding is not None:
            mapping["embedding"] = json.dumps(record.embedding)
        if record.embedding_model is not None:
            mapping["embedding_model"] = record.embedding_model
        return mapping

    def _hash_to_record(self, data: dict[str, str]) -> MemoryRecord:
        """Reconstruct a *MemoryRecord* from a Redis hash mapping."""
        raw: dict[str, Any] = {}
        for key, value in data.items():
            if key in _JSON_FIELDS:
                raw[key] = json.loads(value)
            elif key in ("importance_score", "decay_factor"):
                raw[key] = float(value)
            elif key == "access_count":
                raw[key] = int(value)
            else:
                raw[key] = value
        # Ensure optional fields that were not stored are set to None.
        for opt in _OPTIONAL_FIELDS:
            raw.setdefault(opt, None)
        return MemoryRecord.from_dict(raw)

    def _event_to_json(self, event: MemoryTimelineEvent) -> str:
        return json.dumps(
            {
                "operation": event.operation,
                "memory_id": event.memory_id,
                "timestamp": event.timestamp.isoformat(),
                "actor": event.actor,
                "details": event.details,
            }
        )

    def _json_to_event(self, raw: str) -> MemoryTimelineEvent:
        data = json.loads(raw)
        return MemoryTimelineEvent(
            operation=data["operation"],
            memory_id=data["memory_id"],
            timestamp=parse_datetime(data["timestamp"]) or utcnow(),
            actor=data.get("actor", "system"),
            details=data.get("details", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _all_records(self) -> list[MemoryRecord]:
        ids = await self._redis.smembers(self._ids_key)
        records: list[MemoryRecord] = []
        for memory_id in ids:
            data = await self._redis.hgetall(self._mem_key(memory_id))
            if data:
                records.append(self._hash_to_record(data))
        return records

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    # ------------------------------------------------------------------
    # BaseDriver implementation
    # ------------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        mapping = self._record_to_hash(stored)
        key = self._mem_key(stored.id)
        await self._redis.hset(key, mapping=mapping)
        await self._redis.sadd(self._ids_key, stored.id)
        event = MemoryTimelineEvent(
            operation="upsert",
            memory_id=stored.id,
            timestamp=stored.timestamp_updated,
            details=self._snapshot(stored),
        )
        await self._redis.rpush(self._events_key, self._event_to_json(event))
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[ScoredMemory]:
        records = await self._all_records()
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        records = await self._all_records()
        matched = [r for r in records if record_matches_filters(r, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        data = await self._redis.hgetall(self._mem_key(memory_id))
        if not data:
            raise KeyError(memory_id)
        record = self._hash_to_record(data).touch()
        record.access_count += 1
        record.last_accessed = utcnow()
        # Persist the updated access metadata back to Redis.
        await self._redis.hset(self._mem_key(memory_id), mapping=self._record_to_hash(record))
        event = MemoryTimelineEvent(
            operation="get",
            memory_id=memory_id,
            timestamp=record.last_accessed,
            details=self._snapshot(record),
        )
        await self._redis.rpush(self._events_key, self._event_to_json(event))
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        data = await self._redis.hgetall(self._mem_key(memory_id))
        if not data:
            raise KeyError(memory_id)
        payload = self._hash_to_record(data).to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)
        await self._redis.hset(self._mem_key(memory_id), mapping=self._record_to_hash(updated))
        event = MemoryTimelineEvent(
            operation="update",
            memory_id=memory_id,
            timestamp=updated.timestamp_updated,
            details=self._snapshot(updated),
        )
        await self._redis.rpush(self._events_key, self._event_to_json(event))
        return updated

    async def delete(self, filters: dict[str, Any]) -> int:
        records = await self._all_records()
        ids = [r.id for r in records if record_matches_filters(r, filters)]
        for memory_id in ids:
            # Grab snapshot before deletion for the timeline event.
            data = await self._redis.hgetall(self._mem_key(memory_id))
            snapshot = self._hash_to_record(data) if data else None
            await self._redis.delete(self._mem_key(memory_id))
            await self._redis.srem(self._ids_key, memory_id)
            event = MemoryTimelineEvent(
                operation="delete",
                memory_id=memory_id,
                timestamp=utcnow(),
                details=self._snapshot(snapshot),
            )
            await self._redis.rpush(self._events_key, self._event_to_json(event))
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
        raw_events = await self._redis.lrange(self._events_key, 0, -1)
        filtered: list[MemoryTimelineEvent] = []
        for raw in raw_events:
            event = self._json_to_event(raw)
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
            filtered.append(event)
        return sorted(filtered, key=lambda item: item.timestamp)

    async def explain(self, query: str, filters: dict[str, Any], top_k: int = 5) -> MemoryTrace:
        records = await self._all_records()
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
