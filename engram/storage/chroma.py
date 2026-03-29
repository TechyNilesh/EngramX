from __future__ import annotations

import importlib.util
import json
from typing import Any

from ..embedding import embed_record, embed_text
from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("chromadb",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


# -- helpers for flattening / unflattening metadata --

_LIST_FIELDS = ("provenance", "derived_from", "access_roles")
_DICT_FIELDS = ("payload",)
_EMBEDDING_FIELD = "embedding"

# Chroma metadata values must be str | int | float | bool.  Complex types
# (lists, dicts, None-ish datetimes) are serialised to JSON strings so they
# survive the round-trip.


def _flatten_metadata(record: MemoryRecord) -> dict[str, Any]:
    """Convert a MemoryRecord into a flat dict suitable for Chroma metadata."""
    data = record.to_dict()
    # content is stored as the Chroma *document*, not metadata
    data.pop("content", None)
    # embedding is stored via Chroma's own embedding column
    data.pop("embedding", None)
    data.pop("embedding_model", None)

    meta: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            # Chroma does not accept None – store as empty string sentinel
            meta[key] = ""
            continue
        if key in _LIST_FIELDS or key in _DICT_FIELDS:
            meta[key] = json.dumps(value)
            continue
        if isinstance(value, (list, dict)):
            meta[key] = json.dumps(value)
            continue
        if isinstance(value, bool):
            meta[key] = value
            continue
        if isinstance(value, (int, float)):
            meta[key] = value
            continue
        # everything else as string (datetimes already iso-formatted by to_dict)
        meta[key] = str(value)
    return meta


def _unflatten_metadata(meta: dict[str, Any], document: str) -> dict[str, Any]:
    """Reconstruct a MemoryRecord-compatible dict from Chroma metadata + document."""
    data: dict[str, Any] = {}
    for key, value in meta.items():
        if value == "":
            data[key] = None
            continue
        if key in _LIST_FIELDS or key in _DICT_FIELDS:
            try:
                data[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                data[key] = value
            continue
        if isinstance(value, str) and key not in (
            "id",
            "type",
            "scope",
            "agent_id",
            "user_id",
            "session_id",
            "source",
            "sensitivity_level",
            "retention_policy",
        ):
            # Try to parse JSON for any unknown string field that might be
            # a serialised complex value
            try:
                parsed = json.loads(value)
                if isinstance(parsed, (list, dict)):
                    data[key] = parsed
                    continue
            except (json.JSONDecodeError, TypeError):
                pass
        data[key] = value
    data["content"] = document
    return data


class ChromaDriver(BaseDriver):
    """Storage driver backed by ChromaDB.

    Parameters
    ----------
    path:
        Filesystem path for persistent storage.  ``None`` uses an ephemeral
        in-memory Chroma client.
    collection_name:
        Name of the Chroma collection to use.
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "engram_memories",
    ) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "ChromaDriver requires the optional `chromadb` dependency. "
                "Install the Chroma extra before constructing this driver."
            )

        import chromadb  # type: ignore[import-untyped]

        if path is None:
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=path)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name
        self.path = path
        # Timeline events are tracked in-memory (same pattern as InMemoryDriver)
        self._events: list[MemoryTimelineEvent] = []

    # -- internal helpers -----------------------------------------------------

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    def _chroma_upsert(self, record: MemoryRecord) -> None:
        """Write a single MemoryRecord into the Chroma collection."""
        embedding = embed_record(record)
        meta = _flatten_metadata(record)
        self._collection.upsert(
            ids=[record.id],
            documents=[record.content],
            metadatas=[meta],
            embeddings=[embedding],
        )

    def _chroma_get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single record by id, returning None if missing."""
        result = self._collection.get(
            ids=[memory_id],
            include=["documents", "metadatas", "embeddings"],
        )
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]  # type: ignore[index]
        doc = result["documents"][0]  # type: ignore[index]
        emb = result["embeddings"][0] if result.get("embeddings") else None  # type: ignore[index]
        data = _unflatten_metadata(meta, doc)
        if emb is not None:
            data["embedding"] = list(emb)
        return MemoryRecord.from_dict(data)

    def _all_records(self) -> list[MemoryRecord]:
        """Return every record currently in the collection."""
        result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
        )
        records: list[MemoryRecord] = []
        for i, doc_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]  # type: ignore[index]
            doc = result["documents"][i]  # type: ignore[index]
            emb = result["embeddings"][i] if result.get("embeddings") else None  # type: ignore[index]
            data = _unflatten_metadata(meta, doc)
            if emb is not None:
                data["embedding"] = list(emb)
            records.append(MemoryRecord.from_dict(data))
        return records

    # -- BaseDriver interface -------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        self._chroma_upsert(stored)
        self._events.append(
            MemoryTimelineEvent(
                operation="upsert",
                memory_id=stored.id,
                timestamp=stored.timestamp_updated,
                details=self._snapshot(stored),
            )
        )
        return stored.id

    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[ScoredMemory]:
        records = self._all_records()
        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        records = self._all_records()
        matched = [r for r in records if record_matches_filters(r, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        record = self._chroma_get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        accessed = record.touch()
        accessed.access_count += 1
        accessed.last_accessed = utcnow()
        self._chroma_upsert(accessed)
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
        record = self._chroma_get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        payload = record.to_dict()
        payload.update(patch)
        payload["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(payload)
        self._chroma_upsert(updated)
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
        records = self._all_records()
        ids = [r.id for r in records if record_matches_filters(r, filters)]
        for memory_id in ids:
            existing = self._chroma_get(memory_id)
            self._collection.delete(ids=[memory_id])
            self._events.append(
                MemoryTimelineEvent(
                    operation="delete",
                    memory_id=memory_id,
                    timestamp=utcnow(),
                    details=self._snapshot(existing),
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
            record = MemoryRecord.from_dict(payload) if payload else self._chroma_get(event.memory_id)
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
        records = self._all_records()
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
