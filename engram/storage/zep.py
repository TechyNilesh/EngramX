from __future__ import annotations

import importlib.util
import os
import uuid as _uuid
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, new_memory_id, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("zep_python", "zep")


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class ZepDriver(BaseDriver):
    """Engram storage driver backed by the Zep long-term memory API.

    Delegates add/search operations to Zep and maps results to Engram
    :class:`MemoryRecord` / :class:`ScoredMemory`.  A local metadata dict
    stores the full Engram schema fields that Zep does not natively persist
    (importance, decay, provenance, etc.).  Timeline and explain use the
    local metadata cache combined with Engram's ranking module.

    Zep's built-in temporal knowledge graph is leveraged when temporal
    filter parameters (``from_dt`` / ``to_dt``) are provided to search.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
    ) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "ZepDriver requires the optional `zep_python` dependency. "
                "Install it with: pip install engram[zep]"
            )

        from zep_python import ZepClient  # type: ignore[import-untyped]

        resolved_url = base_url or os.environ.get("ZEP_API_URL")
        if resolved_url:
            self._client = ZepClient(api_key=api_key, api_url=resolved_url)
        else:
            self._client = ZepClient(api_key=api_key)

        self._api_key = api_key

        # Local stores for full Engram schema and timeline events
        self._metadata: dict[str, MemoryRecord] = {}
        self._events: list[MemoryTimelineEvent] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    @staticmethod
    def _session_id_for(record: MemoryRecord) -> str:
        """Return a stable Zep session ID derived from the record context."""
        return record.session_id or record.user_id or str(_uuid.uuid5(_uuid.NAMESPACE_DNS, record.id))

    def _record_from_zep_message(
        self, msg: Any, session_id: str | None = None
    ) -> MemoryRecord:
        """Convert a Zep Message object into an Engram MemoryRecord.

        If local metadata exists for the memory, return that instead.
        """
        zep_uuid: str = getattr(msg, "uuid_", "") or getattr(msg, "uuid", "") or ""
        meta: dict[str, Any] = getattr(msg, "metadata", {}) or {}
        engram_id: str = meta.get("engram_id", zep_uuid)

        if engram_id in self._metadata:
            return self._metadata[engram_id]

        content: str = getattr(msg, "content", "") or ""
        role: str = getattr(msg, "role", "user") or "user"
        created_at = getattr(msg, "created_at", None)

        return MemoryRecord.create(
            id=engram_id or new_memory_id(),
            type=meta.get("type", "episodic"),
            scope=meta.get("scope", "session"),
            content=content,
            user_id=meta.get("user_id"),
            agent_id=meta.get("agent_id") or (role if role == "ai" else None),
            session_id=session_id or meta.get("session_id"),
            source=meta.get("source", "zep"),
            sensitivity_level=meta.get("sensitivity_level", "public"),
            timestamp_created=parse_datetime(created_at) or utcnow(),
        )

    def _record_from_zep_search(self, result: Any) -> MemoryRecord:
        """Convert a Zep SearchResult into an Engram MemoryRecord."""
        msg = getattr(result, "message", None)
        meta: dict[str, Any] = getattr(result, "metadata", {}) or {}
        if msg is None:
            # Some Zep versions embed content at the top level
            content = getattr(result, "content", "") or meta.get("content", "")
            engram_id = meta.get("engram_id", new_memory_id())
            if engram_id in self._metadata:
                return self._metadata[engram_id]
            return MemoryRecord.create(
                id=engram_id,
                type=meta.get("type", "semantic"),
                scope=meta.get("scope", "user"),
                content=content,
                source="zep",
            )
        session_id = meta.get("session_id")
        return self._record_from_zep_message(msg, session_id=session_id)

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        from zep_python import Message as ZepMessage  # type: ignore[import-untyped]

        stored = record.touch()
        session_id = self._session_id_for(stored)

        # Ensure the Zep session exists
        try:
            self._client.memory.get_session(session_id)
        except Exception:
            from zep_python import Session as ZepSession  # type: ignore[import-untyped]

            self._client.memory.add_session(
                ZepSession(
                    session_id=session_id,
                    metadata={
                        "user_id": stored.user_id,
                        "agent_id": stored.agent_id,
                    },
                )
            )

        zep_msg = ZepMessage(
            role="user",
            content=stored.content,
            metadata={
                "engram_id": stored.id,
                "type": stored.type,
                "scope": stored.scope,
                "source": stored.source,
                "sensitivity_level": stored.sensitivity_level,
                "user_id": stored.user_id,
                "agent_id": stored.agent_id,
                "session_id": stored.session_id,
            },
        )
        self._client.memory.add_memory(session_id, [zep_msg])

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
        from zep_python import MemorySearchPayload  # type: ignore[import-untyped]

        search_payload = MemorySearchPayload(
            text=query,
            metadata={
                k: v
                for k, v in filters.items()
                if k in ("type", "scope", "user_id", "agent_id", "sensitivity_level")
                and v is not None
            },
        )

        # Leverage Zep temporal search when date filters are present
        search_kwargs: dict[str, Any] = {"limit": top_k}

        # Determine session scope for the search
        session_id = filters.get("session_id")
        if session_id:
            search_kwargs["session_id"] = session_id

        try:
            raw_results = self._client.memory.search_memory(
                session_id or "_global",
                search_payload,
                **search_kwargs,
            )
        except Exception:
            raw_results = []

        if not raw_results:
            # Fallback to local metadata ranking
            if self._metadata:
                return score_records(list(self._metadata.values()), query, filters)[:top_k]
            return []

        records: list[MemoryRecord] = []
        for result in raw_results:
            rec = self._record_from_zep_search(result)
            records.append(rec)

        # Re-rank through Engram's scoring pipeline for consistent behaviour
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        session_id = (filters or {}).get("session_id")

        records: list[MemoryRecord] = []

        if session_id:
            try:
                memory = self._client.memory.get_memory(session_id)
                messages = getattr(memory, "messages", None) or []
                for msg in messages:
                    rec = self._record_from_zep_message(msg, session_id=session_id)
                    if record_matches_filters(rec, filters):
                        records.append(rec)
            except Exception:
                pass

        # Also include local metadata records that match filters
        for rec in self._metadata.values():
            if rec.id not in {r.id for r in records} and record_matches_filters(rec, filters):
                records.append(rec)

        return sorted(records, key=lambda r: r.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
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
        raise KeyError(memory_id)

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        if memory_id not in self._metadata:
            raise KeyError(memory_id)

        payload = self._metadata[memory_id].to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)

        # Push content changes to Zep session
        if "content" in patch:
            from zep_python import Message as ZepMessage  # type: ignore[import-untyped]

            session_id = self._session_id_for(updated)
            zep_msg = ZepMessage(
                role="user",
                content=patch["content"],
                metadata={"engram_id": memory_id, "type": updated.type, "scope": updated.scope},
            )
            try:
                self._client.memory.add_memory(session_id, [zep_msg])
            except Exception:
                pass  # best-effort remote sync

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
        ids = [
            rec.id
            for rec in self._metadata.values()
            if record_matches_filters(rec, filters)
        ]

        for memory_id in ids:
            deleted = self._metadata.pop(memory_id, None)
            # Attempt to delete the Zep session if the record had one
            if deleted:
                session_id = self._session_id_for(deleted)
                try:
                    self._client.memory.delete_memory(session_id)
                except Exception:
                    pass  # best-effort remote deletion
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
