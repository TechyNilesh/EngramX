from __future__ import annotations

import pytest

from engram.adapters import EngramChatAdapter
from engram.client import MemoryClient
from engram.storage.memory import InMemoryDriver


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

    async def responder(messages):
        captured["messages"] = messages
        return {"choices": [{"message": {"content": "Use metric units."}}]}

    adapter = EngramChatAdapter(client=client, responder=responder)
    response, attribution = await adapter.run_with_attribution("What units should I use?", user_id="u-1")

    assert response == "Use metric units."
    assert attribution.memories_used
    assert captured["messages"][0]["role"] == "system"
    assert "User prefers metric units." in captured["messages"][0]["content"]
