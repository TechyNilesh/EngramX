"""LangChain adapter for Engram memory."""

from __future__ import annotations

from typing import Any

from ..client import MemoryClient

try:
    from langchain.memory import BaseChatMemory

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

if _HAS_LANGCHAIN:

    class EngramChatMemory(BaseChatMemory):  # type: ignore[misc]
        """LangChain chat memory backed by Engram.

        Implements LangChain's ``BaseChatMemory`` interface so it can be used as
        a drop-in replacement for ``ConversationBufferMemory`` and similar classes.
        """

        client: Any  # MemoryClient (Any to satisfy pydantic)
        user_id: str | None = None
        memory_key: str = "history"
        top_k: int = 5

        class Config:
            arbitrary_types_allowed = True

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
            """Search Engram for memories relevant to the current input."""
            query = inputs.get("input") or inputs.get("question") or ""
            if not query:
                # Fall back to the first string value in inputs
                for value in inputs.values():
                    if isinstance(value, str):
                        query = value
                        break

            if not query:
                return {self.memory_key: ""}

            filters: dict[str, Any] = {}
            if self.user_id is not None:
                filters["user_id"] = self.user_id

            results = self.client.search_sync(query, filters=filters, top_k=self.top_k)
            formatted = "\n".join(
                f"[{item.record.type}] {item.record.content}" for item in results
            )
            return {self.memory_key: formatted}

        def save_context(
            self, inputs: dict[str, Any], outputs: dict[str, str]
        ) -> None:
            """Store a conversation turn as an episodic memory in Engram."""
            input_str = inputs.get("input") or inputs.get("question") or ""
            output_str = outputs.get("output") or outputs.get("response") or ""
            if not input_str and not output_str:
                return

            content = f"Human: {input_str}\nAssistant: {output_str}"
            self.client.add_sync(
                type="episodic",
                scope="user" if self.user_id else "session",
                user_id=self.user_id,
                content=content,
                source="langchain",
            )

        def clear(self) -> None:
            """Delete memories for this user."""
            from ..client import _run_sync

            filters: dict[str, Any] = {}
            if self.user_id is not None:
                filters["user_id"] = self.user_id
            _run_sync(self.client.delete(filters))

else:
    # Standalone class with the same interface when langchain is not installed.
    class EngramChatMemory:  # type: ignore[no-redef]
        """Standalone Engram chat memory with a LangChain-compatible interface.

        This class is used when LangChain is not installed.  It exposes the same
        public API so downstream code can rely on it without importing LangChain.
        """

        def __init__(
            self,
            client: MemoryClient,
            user_id: str | None = None,
            memory_key: str = "history",
            top_k: int = 5,
        ) -> None:
            self.client = client
            self.user_id = user_id
            self.memory_key = memory_key
            self.top_k = top_k

        @property
        def memory_variables(self) -> list[str]:
            return [self.memory_key]

        def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
            """Search Engram for memories relevant to the current input."""
            query = inputs.get("input") or inputs.get("question") or ""
            if not query:
                for value in inputs.values():
                    if isinstance(value, str):
                        query = value
                        break

            if not query:
                return {self.memory_key: ""}

            filters: dict[str, Any] = {}
            if self.user_id is not None:
                filters["user_id"] = self.user_id

            results = self.client.search_sync(query, filters=filters, top_k=self.top_k)
            formatted = "\n".join(
                f"[{item.record.type}] {item.record.content}" for item in results
            )
            return {self.memory_key: formatted}

        def save_context(
            self, inputs: dict[str, Any], outputs: dict[str, str]
        ) -> None:
            """Store a conversation turn as an episodic memory in Engram."""
            input_str = inputs.get("input") or inputs.get("question") or ""
            output_str = outputs.get("output") or outputs.get("response") or ""
            if not input_str and not output_str:
                return

            content = f"Human: {input_str}\nAssistant: {output_str}"
            self.client.add_sync(
                type="episodic",
                scope="user" if self.user_id else "session",
                user_id=self.user_id,
                content=content,
                source="langchain",
            )

        def clear(self) -> None:
            """Delete memories for this user."""
            from ..client import _run_sync

            filters: dict[str, Any] = {}
            if self.user_id is not None:
                filters["user_id"] = self.user_id
            _run_sync(self.client.delete(filters))
