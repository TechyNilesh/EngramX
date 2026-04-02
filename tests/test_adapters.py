from __future__ import annotations

import pytest

from engramxx.client import MemoryClient
from engramxx.observability import ScoredMemory
from engramxx.storage.memory import InMemoryDriver


@pytest.mark.asyncio
async def test_run_with_attribution_uses_retrieved_memories():
    client = MemoryClient(InMemoryDriver())
    await client.add(
        type="semantic",
        scope="user",
        user_id="u-1",
        content="User prefers metric units.",
        source="conversation",
        importance_score=0.8,
    )

    captured = {}

    async def runner(prompt: str, memories: list[ScoredMemory]) -> str:
        captured["prompt"] = prompt
        captured["memories"] = memories
        return "Use metric units."

    response, attribution = await client.run_with_attribution(
        "What units should I use?",
        user_id="u-1",
        runner=runner,
    )

    assert response == "Use metric units."
    assert attribution.memories_used
    assert "User prefers metric units." in captured["prompt"]
