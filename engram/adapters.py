from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping

from .client import MemoryClient
from .models import MemorySignal
from .observability import OutputAttribution, ScoredMemory


Responder = Callable[[list[dict[str, str]]], Any]


def _extract_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("content", "text", "message"):
            value = response.get(key)
            if isinstance(value, str):
                return value
        choices = response.get("choices")
        if choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                content = first.get("content")
                if isinstance(content, str):
                    return content
    choices = getattr(response, "choices", None)
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
        content = getattr(first, "content", None)
        if isinstance(content, str):
            return content
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return str(response)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(slots=True)
class EngramChatAdapter:
    client: MemoryClient
    responder: Responder
    system_prompt: str = "You are a helpful AI. Answer using the user query and memories."
    top_k: int = 3

    def build_messages(self, task: str, memories: Iterable[ScoredMemory]) -> list[dict[str, str]]:
        memories_text = "\n".join(f"- {item.record.content}" for item in memories) or "- None yet."
        system_message = f"{self.system_prompt}\nUser Memories:\n{memories_text}"
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": task},
        ]

    async def run_with_attribution(
        self,
        task: str,
        *,
        user_id: str = "default_user",
        session_id: str | None = None,
        memory_filters: dict[str, Any] | None = None,
    ) -> tuple[str, OutputAttribution]:
        filters = {"user_id": user_id}
        if memory_filters:
            filters.update(memory_filters)

        memories = await self.client.search(task, filters=filters, top_k=self.top_k)
        messages = self.build_messages(task, memories)
        response = await _maybe_await(self.responder(messages))
        assistant_text = _extract_text(response)

        await self.client.add(
            type="episodic",
            scope="user",
            user_id=user_id,
            session_id=session_id,
            content=f"User: {task}\nAssistant: {assistant_text}",
            source="conversation",
            importance_score=0.7,
            sensitivity_level="internal",
        )

        policies_fired: list[str] = []
        if self.client.policy_engine is not None:
            outcome = await self.client.ingest_event(
                MemorySignal(
                    type="conversation_turn",
                    content=task,
                    source="conversation",
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"task": task},
                )
            )
            policies_fired.extend(outcome.policies_fired)

        attribution = self.client.build_attribution(memories, policies_fired=policies_fired)
        return assistant_text, attribution
