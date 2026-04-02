"""AutoGen adapter for Engram memory."""

from __future__ import annotations

from typing import Any

from ..client import MemoryClient, _run_sync


class EngramAutoGenMemory:
    """Integrates Engram with AutoGen's memory interface.

    Provides ``query``, ``add``, ``clear``, and ``update_context`` methods so
    that AutoGen agents can transparently read and write long-term memories
    through Engram.
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

    def query(self, query: str) -> list[dict[str, Any]]:
        """Search Engram and return results as a list of dicts."""
        results = self.client.search_sync(
            query, filters=self._build_filters(), top_k=self.top_k
        )
        return [
            {
                "id": item.record.id,
                "content": item.record.content,
                "type": item.record.type,
                "score": item.score,
            }
            for item in results
        ]

    def add(self, content: str, **kwargs: Any) -> str:
        """Store a memory in Engram.

        Extra keyword arguments are forwarded to :meth:`MemoryClient.add_sync`,
        allowing callers to override fields like ``type`` or ``scope``.
        """
        defaults: dict[str, Any] = {
            "type": kwargs.pop("type", "episodic"),
            "scope": kwargs.pop("scope", "user" if self.user_id else "session"),
            "user_id": kwargs.pop("user_id", self.user_id),
            "content": content,
            "source": kwargs.pop("source", "autogen"),
        }
        defaults.update(kwargs)
        return self.client.add_sync(**defaults)

    def clear(self) -> int:
        """Delete all memories matching the current filters.

        Returns the number of deleted records.
        """
        return _run_sync(self.client.delete(self._build_filters()))

    def update_context(
        self,
        agent: Any,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Inject relevant memories into an AutoGen agent's message context.

        Extracts the latest user message, searches Engram for related memories,
        and prepends a system message containing those memories to the message
        list.  Returns the augmented message list.
        """
        # Find the most recent user message to use as a search query.
        query = ""
        for message in reversed(messages):
            if message.get("role") == "user" and message.get("content"):
                query = message["content"]
                break

        if not query:
            return messages

        results = self.client.search_sync(
            query, filters=self._build_filters(), top_k=self.top_k
        )

        if not results:
            return messages

        memory_text = "\n".join(
            f"- [{item.record.type}] {item.record.content}" for item in results
        )
        memory_message: dict[str, Any] = {
            "role": "system",
            "content": f"Relevant memories:\n{memory_text}",
        }

        # Prepend the memory message right after any existing system messages.
        insert_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
            else:
                break

        augmented = list(messages)
        augmented.insert(insert_idx, memory_message)
        return augmented
