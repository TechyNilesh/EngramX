from __future__ import annotations

import importlib.util
import os
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, new_memory_id, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("mem0",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class Mem0Driver(BaseDriver):
    """Engram storage driver backed by the Mem0 managed-memory API.

    Delegates core add/search operations to Mem0 while maintaining a local
    metadata dictionary for the full :class:`MemoryRecord` schema fields that
    Mem0 does not natively store (importance, decay, provenance, etc.).
    Timeline and explain are handled locally using the metadata cache.
    """

    def __init__(
        self,
        api_key: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "Mem0Driver requires the optional `mem0` dependency. "
                "Install it with: pip install engramx[mem0]"
            )

        from mem0 import MemoryClient as Mem0MemoryClient  # type: ignore[import-untyped]

        resolved_key = api_key or os.environ.get("MEM0_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "A Mem0 API key is required. Pass api_key= or set the MEM0_API_KEY env var."
            )

        self._client = Mem0MemoryClient(api_key=resolved_key)
        self._org_id = org_id
        self._project_id = project_id

        # Local stores for full Engram schema and timeline events
        self._metadata: dict[str, MemoryRecord] = {}
        self._events: list[MemoryTimelineEvent] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    def _build_mem0_kwargs(self, record: MemoryRecord) -> dict[str, Any]:
        """Build keyword arguments for the Mem0 add() call."""
        kwargs: dict[str, Any] = {}
        if record.user_id:
            kwargs["user_id"] = record.user_id
        if record.agent_id:
            kwargs["agent_id"] = record.agent_id
        if record.session_id:
            kwargs["run_id"] = record.session_id
        if self._org_id:
            kwargs["org_id"] = self._org_id
        if self._project_id:
            kwargs["project_id"] = self._project_id
        metadata: dict[str, Any] = {
            "engram_id": record.id,
            "type": record.type,
            "scope": record.scope,
            "source": record.source,
            "sensitivity_level": record.sensitivity_level,
        }
        kwargs["metadata"] = metadata
        return kwargs

    def _record_from_mem0(self, mem0_item: dict[str, Any]) -> MemoryRecord:
        """Convert a Mem0 API result dict back into an Engram MemoryRecord.

        If we have local metadata for this memory, merge it; otherwise construct
        a minimal record from whatever Mem0 returns.
        """
        mem0_id: str = mem0_item.get("id", "")
        mem0_meta: dict[str, Any] = mem0_item.get("metadata", {}) or {}
        engram_id: str = mem0_meta.get("engram_id", mem0_id)

        # Prefer the full local record if available
        if engram_id in self._metadata:
            return self._metadata[engram_id]

        # Fallback: build a record from Mem0 data
        content: str = mem0_item.get("memory", "") or mem0_item.get("text", "")
        return MemoryRecord.create(
            id=engram_id,
            type=mem0_meta.get("type", "semantic"),
            scope=mem0_meta.get("scope", "user"),
            content=content,
            user_id=mem0_item.get("user_id"),
            agent_id=mem0_item.get("agent_id"),
            session_id=mem0_item.get("run_id"),
            source=mem0_meta.get("source", "mem0"),
            sensitivity_level=mem0_meta.get("sensitivity_level", "public"),
        )

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        kwargs = self._build_mem0_kwargs(stored)

        # Mem0 add() expects a list of messages
        messages = [{"role": "user", "content": stored.content}]
        self._client.add(messages, **kwargs)

        # Persist full schema locally
        self._metadata[stored.id] = stored
        self._events.append(
            MemoryTimelineEvent(
                operation="upsert",
                memory_id=stored.id,
                timestamp=stored.timestamp_updated,
                details=self._snapshot(stored),
            )
        )
        return stored.id

    async def search(
        self, query: str, filters: dict[str, Any], top_k: int
    ) -> list[ScoredMemory]:
        search_kwargs: dict[str, Any] = {}
        if filters.get("user_id"):
            search_kwargs["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_kwargs["agent_id"] = filters["agent_id"]
        if filters.get("session_id"):
            search_kwargs["run_id"] = filters["session_id"]
        if self._org_id:
            search_kwargs["org_id"] = self._org_id
        if self._project_id:
            search_kwargs["project_id"] = self._project_id

        raw_results: list[dict[str, Any]] = self._client.search(query, **search_kwargs) or []

        # Convert Mem0 results to Engram ScoredMemory via local ranking
        records: list[MemoryRecord] = []
        for idx, item in enumerate(raw_results):
            rec = self._record_from_mem0(item)
            records.append(rec)

        # If we have records with local metadata, use Engram's ranking
        if records:
            return score_records(records, query, filters)[:top_k]
        return []

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        # Try to get all memories from Mem0 for the user/agent scope
        list_kwargs: dict[str, Any] = {}
        if filters and filters.get("user_id"):
            list_kwargs["user_id"] = filters["user_id"]
        if filters and filters.get("agent_id"):
            list_kwargs["agent_id"] = filters["agent_id"]
        if filters and filters.get("session_id"):
            list_kwargs["run_id"] = filters["session_id"]

        raw: list[dict[str, Any]] = self._client.get_all(**list_kwargs) or []

        records: list[MemoryRecord] = []
        for item in raw:
            rec = self._record_from_mem0(item)
            if record_matches_filters(rec, filters):
                records.append(rec)

        return sorted(records, key=lambda r: r.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        # Check local metadata first
        if memory_id in self._metadata:
            record = self._metadata[memory_id]
            accessed = record.touch()
            accessed.access_count += 1
            accessed.last_accessed = utcnow()
            self._metadata[memory_id] = accessed
            self._events.append(
                MemoryTimelineEvent(
                    operation="get",
                    memory_id=memory_id,
                    timestamp=accessed.last_accessed,
                    details=self._snapshot(accessed),
                )
            )
            return accessed

        # Fallback: try Mem0 API
        try:
            raw: dict[str, Any] = self._client.get(memory_id)
        except Exception:
            raise KeyError(memory_id)

        if not raw:
            raise KeyError(memory_id)

        record = self._record_from_mem0(raw)
        record.access_count += 1
        record.last_accessed = utcnow()
        self._metadata[record.id] = record
        self._events.append(
            MemoryTimelineEvent(
                operation="get",
                memory_id=record.id,
                timestamp=record.last_accessed,
                details=self._snapshot(record),
            )
        )
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        if memory_id not in self._metadata:
            # Ensure we have a local record to patch
            await self.get(memory_id)

        payload = self._metadata[memory_id].to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)

        # Push content changes to Mem0
        if "content" in patch:
            self._client.update(memory_id, data=patch["content"])

        self._metadata[memory_id] = updated
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
        # Identify matching records from local metadata
        ids = [
            rec.id
            for rec in self._metadata.values()
            if record_matches_filters(rec, filters)
        ]

        for memory_id in ids:
            try:
                self._client.delete(memory_id)
            except Exception:
                pass  # best-effort remote deletion
            deleted = self._metadata.pop(memory_id, None)
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
        start = parse_datetime(from_dt)
        end = parse_datetime(to_dt)
        filtered: list[MemoryTimelineEvent] = []

        for event in self._events:
            payload = event.details.get("record")
            record = (
                MemoryRecord.from_dict(payload)
                if payload
                else self._metadata.get(event.memory_id)
            )
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

    async def explain(
        self, query: str, filters: dict[str, Any], top_k: int = 5
    ) -> MemoryTrace:
        scored = score_records(list(self._metadata.values()), query, filters)
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
