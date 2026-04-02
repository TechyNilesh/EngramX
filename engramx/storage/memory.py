from __future__ import annotations

from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory
from ..schema import MemoryRecord, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records


class InMemoryDriver(BaseDriver):
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._events: list[MemoryTimelineEvent] = []

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        self._records[stored.id] = stored
        self._events.append(
            MemoryTimelineEvent(
                operation="upsert",
                memory_id=stored.id,
                timestamp=stored.timestamp_updated,
                details=self._snapshot(stored),
            )
        )
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[RetrievedMemory]:
        return score_records(list(self._records.values()), query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        records = [record for record in self._records.values() if record_matches_filters(record, filters)]
        return sorted(records, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        if memory_id not in self._records:
            raise KeyError(memory_id)
        record = self._records[memory_id]
        accessed = record.touch()
        accessed.access_count += 1
        accessed.last_accessed = utcnow()
        self._records[memory_id] = accessed
        self._events.append(
            MemoryTimelineEvent(
                operation="get",
                memory_id=memory_id,
                timestamp=accessed.last_accessed,
                details=self._snapshot(accessed),
            )
        )
        return accessed

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        if memory_id not in self._records:
            raise KeyError(memory_id)
        payload = self._records[memory_id].to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)
        self._records[memory_id] = updated
        self._events.append(
            MemoryTimelineEvent(
                operation="update",
                memory_id=memory_id,
                timestamp=updated.timestamp_updated,
                details=self._snapshot(updated),
            )
        )
        return updated

    async def delete(self, filters: dict[str, Any]) -> int:
        ids = [record.id for record in self._records.values() if record_matches_filters(record, filters)]
        for memory_id in ids:
            deleted = self._records.pop(memory_id)
            self._events.append(
                MemoryTimelineEvent(
                    operation="delete",
                    memory_id=memory_id,
                    timestamp=utcnow(),
                    details=self._snapshot(deleted),
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
        from ..schema import parse_datetime

        start = parse_datetime(from_dt)
        end = parse_datetime(to_dt)
        filtered: list[MemoryTimelineEvent] = []
        for event in self._events:
            payload = event.details.get("record")
            record = MemoryRecord.from_dict(payload) if payload else self._records.get(event.memory_id)
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
        scored = score_records(list(self._records.values()), query, filters)
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
