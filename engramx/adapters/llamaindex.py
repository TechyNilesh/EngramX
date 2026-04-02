"""LlamaIndex adapter for Engram memory."""

from __future__ import annotations

from typing import Any

from ..client import MemoryClient, _run_sync


class EngramMemoryBlock:
    """Wraps :class:`MemoryClient` for use with LlamaIndex agents.

    Provides a simple ``get`` / ``put`` interface that LlamaIndex memory
    blocks expect, backed by Engram's search and storage capabilities.
    """

    def __init__(
        self,
        client: MemoryClient,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> None:
        self.client = client
        self.user_id = user_id
        self.top_k = top_k

    def _build_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.user_id is not None:
            filters["user_id"] = self.user_id
        return filters

    def get(self, query: str) -> str:
        """Search memories and return a formatted string of results."""
        results = self.client.search_sync(
            query, filters=self._build_filters(), top_k=self.top_k
        )
        if not results:
            return ""
        return "\n".join(
            f"- [{item.record.type}] {item.record.content}" for item in results
        )

    def put(self, content: str) -> str:
        """Store *content* as a semantic memory and return the new memory ID."""
        return self.client.add_sync(
            type="semantic",
            scope="user" if self.user_id else "session",
            user_id=self.user_id,
            content=content,
            source="llamaindex",
        )

    def get_all(self) -> list[dict[str, Any]]:
        """Return all memories matching the current filters as dicts."""
        records = _run_sync(self.client.list(filters=self._build_filters()))
        return [record.to_dict() for record in records]

    def reset(self) -> None:
        """Clear all memories matching the current filters."""
        _run_sync(self.client.delete(self._build_filters()))
