from __future__ import annotations

from typing import Any, Iterable, Mapping

from .client import MemoryClient
from .models import MemorySignal


class MemoryHarness:
    def __init__(
        self,
        driver: str = "sqlite",
        config: Mapping[str, Any] | str | None = None,
    ) -> None:
        if config is None:
            config = {
                "driver": {"kind": driver, "path": ":memory:"},
                "policies": {
                    "extraction": [
                        {
                            "name": "extract-user-preferences",
                            "trigger": "conversation_turn",
                            "conditions": [{"content_matches": ["prefer"]}],
                            "create": {
                                "type": "semantic",
                                "scope": "user",
                                "importance_score": 0.8,
                                "sensitivity_level": "internal",
                            },
                        }
                    ]
                },
            }
        self.client = MemoryClient(config=config)
        self._pending_events: list[MemorySignal] = []

    async def inject_conversation(self, messages: Iterable[tuple[str, str]]) -> list[str]:
        ids: list[str] = []
        for index, (role, content) in enumerate(messages):
            user_id = "harness-user" if role == "user" else None
            session_id = "harness-session"
            ids.append(
                await self.client.add(
                    type="episodic",
                    scope="session",
                    user_id=user_id,
                    session_id=session_id,
                    content=content,
                    source="conversation",
                    sensitivity_level="internal",
                    provenance=[role, f"turn-{index}"],
                    payload={"role": role},
                )
            )
            self._pending_events.append(
                MemorySignal(
                    type="conversation_turn",
                    content=content,
                    source="conversation",
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"role": role, "event_id": f"turn-{index}"},
                )
            )
        return ids

    async def run_policies(self) -> None:
        for event in self._pending_events:
            await self.client.ingest_event(event)
        self._pending_events.clear()
        await self.client.promote()
        await self.client.apply_decay()

    async def search(self, query: str, **filters: Any):
        return await self.client.search(query=query, filters=filters)
