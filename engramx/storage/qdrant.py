from __future__ import annotations

import importlib.util
import uuid as _uuid
from typing import Any

from ..embedding import EMBEDDING_DIM, embed_record, embed_text
from ..observability import MemoryTimelineEvent, MemoryTrace, RetrievedMemory, ScoredMemory
from ..schema import MemoryRecord, parse_datetime, utcnow
from .base import BaseDriver
from .ranking import record_matches_filters, score_records

_DEPENDENCIES = ("qdrant_client",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


def _record_to_payload(record: MemoryRecord) -> dict[str, Any]:
    """Convert a MemoryRecord to a Qdrant-compatible payload dict."""
    data = record.to_dict()
    # Remove embedding from payload; it is stored as the point vector.
    data.pop("embedding", None)
    return data


def _payload_to_record(payload: dict[str, Any], vector: list[float] | None = None) -> MemoryRecord:
    """Reconstruct a MemoryRecord from a Qdrant payload and optional vector."""
    data = dict(payload)
    data["embedding"] = vector
    return MemoryRecord.from_dict(data)


def _deterministic_uuid(memory_id: str) -> str:
    """Return a stable UUID-format string derived from the memory id.

    Qdrant requires point ids to be either unsigned integers or UUID strings.
    MemoryRecord ids are already UUIDs in most cases; if not, we derive one
    deterministically so the mapping is repeatable.
    """
    try:
        _uuid.UUID(memory_id)
        return memory_id
    except ValueError:
        return str(_uuid.uuid5(_uuid.NAMESPACE_URL, memory_id))


class QdrantDriver(BaseDriver):
    """Qdrant-backed memory storage driver.

    Parameters
    ----------
    url:
        Qdrant server URL.  When *None* (the default) an **in-memory** Qdrant
        instance is used -- handy for tests and local development.
    collection_name:
        Name of the Qdrant collection that holds memories.
    embedding_dim:
        Dimensionality of the embedding vectors stored alongside each memory.
    """

    def __init__(
        self,
        url: str | None = None,
        collection_name: str = "engram_memories",
        embedding_dim: int = EMBEDDING_DIM,
    ) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "QdrantDriver requires the optional `qdrant-client` dependency. "
                "Install it with:  pip install qdrant-client"
            )

        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        if url is None:
            self._client = QdrantClient(location=":memory:")
        else:
            self._client = QdrantClient(url=url)

        # Create collection if it does not already exist.
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
            )

        # In-memory timeline event log (mirrors the pattern used by other drivers).
        self._events: list[MemoryTimelineEvent] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self, record: MemoryRecord | None) -> dict[str, Any]:
        return {"record": record.to_dict() if record else None}

    def _upsert_point(self, record: MemoryRecord) -> None:
        from qdrant_client.models import PointStruct

        vector = embed_record(record, dims=self.embedding_dim)
        payload = _record_to_payload(record)
        point_id = _deterministic_uuid(record.id)
        self._client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    def _get_point(self, memory_id: str) -> tuple[dict[str, Any], list[float]] | None:
        """Retrieve a single point by memory id.  Returns (payload, vector) or None."""
        point_id = _deterministic_uuid(memory_id)
        results = self._client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_payload=True,
            with_vectors=True,
        )
        if not results:
            return None
        point = results[0]
        vector = list(point.vector) if point.vector else None
        return dict(point.payload), vector

    def _all_records(self) -> list[MemoryRecord]:
        """Scroll through every point in the collection and return MemoryRecords."""
        records: list[MemoryRecord] = []
        offset = None
        while True:
            result = self._client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = result
            for point in points:
                vector = list(point.vector) if point.vector else None
                records.append(_payload_to_record(dict(point.payload), vector))
            if next_offset is None:
                break
            offset = next_offset
        return records

    def _delete_point(self, memory_id: str) -> None:
        from qdrant_client.models import PointIdsList

        point_id = _deterministic_uuid(memory_id)
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[point_id]),
        )

    # ------------------------------------------------------------------
    # BaseDriver interface
    # ------------------------------------------------------------------

    async def upsert(self, record: MemoryRecord) -> str:
        stored = record.touch()
        self._upsert_point(stored)
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
        # Use Qdrant vector search to get a broad candidate set, then apply
        # the framework's own ranking / filtering for consistency with other
        # drivers.
        query_vector = embed_text(query, dims=self.embedding_dim)

        # Request more candidates than top_k so that post-filtering still
        # yields enough results.
        search_limit = max(top_k * 4, 64)
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=search_limit,
            with_payload=True,
            with_vectors=True,
        )

        records: list[MemoryRecord] = []
        for hit in hits:
            vector = list(hit.vector) if hit.vector else None
            records.append(_payload_to_record(dict(hit.payload), vector))

        return score_records(records, query, filters)[:top_k]

    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]:
        records = self._all_records()
        matched = [r for r in records if record_matches_filters(r, filters)]
        return sorted(matched, key=lambda item: item.timestamp_updated, reverse=True)

    async def get(self, memory_id: str) -> MemoryRecord:
        result = self._get_point(memory_id)
        if result is None:
            raise KeyError(memory_id)
        payload, vector = result
        record = _payload_to_record(payload, vector).touch()
        record.access_count += 1
        record.last_accessed = utcnow()
        # Persist the access-count / timestamp update back to Qdrant.
        self._upsert_point(record)
        self._events.append(
            MemoryTimelineEvent(
                operation="get",
                memory_id=memory_id,
                timestamp=record.last_accessed,
                details=self._snapshot(record),
            )
        )
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        result = self._get_point(memory_id)
        if result is None:
            raise KeyError(memory_id)
        payload, vector = result
        existing = _payload_to_record(payload, vector)
        merged = existing.to_dict()
        merged.update(patch)
        merged["timestamp_updated"] = utcnow().isoformat()
        updated = MemoryRecord.from_dict(merged)
        self._upsert_point(updated)
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
        targets = [r for r in records if record_matches_filters(r, filters)]
        for record in targets:
            self._delete_point(record.id)
            self._events.append(
                MemoryTimelineEvent(
                    operation="delete",
                    memory_id=record.id,
                    timestamp=utcnow(),
                    details=self._snapshot(record),
                )
            )
        return len(targets)

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
        # Retrieve a generous candidate set via Qdrant vector search, then
        # score with the shared ranking logic for a full trace.
        query_vector = embed_text(query, dims=self.embedding_dim)
        search_limit = max(top_k * 4, 64)
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=search_limit,
            with_payload=True,
            with_vectors=True,
        )

        records: list[MemoryRecord] = []
        for hit in hits:
            vector = list(hit.vector) if hit.vector else None
            records.append(_payload_to_record(dict(hit.payload), vector))

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
