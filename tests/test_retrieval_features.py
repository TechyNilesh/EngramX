from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from engram.client import MemoryClient
from engram.schema import MemoryRecord, new_memory_id
from engram.storage.memory import InMemoryDriver


@pytest.mark.asyncio
async def test_new_memory_id_looks_like_uuidv7():
    memory_id = new_memory_id()
    parsed = uuid.UUID(memory_id)
    assert parsed.version == 7


@pytest.mark.asyncio
async def test_hybrid_retrieval_with_temporal_filters():
    client = MemoryClient(InMemoryDriver())
    old_dt = datetime(2025, 1, 1, tzinfo=timezone.utc)
    recent_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await client.add(
        record=MemoryRecord(
            id="mem-old",
            type="semantic",
            scope="user",
            user_id="u-1",
            content="User prefers metric units for measurements.",
            timestamp_created=old_dt,
            timestamp_updated=old_dt,
        )
    )
    await client.add(
        record=MemoryRecord(
            id="mem-recent",
            type="semantic",
            scope="user",
            user_id="u-1",
            content="User prefers metric preferences for all reports.",
            timestamp_created=recent_dt,
            timestamp_updated=recent_dt,
        )
    )

    results = await client.search(
        "metric preferences",
        filters={"user_id": "u-1", "created_after": "2025-06-01T00:00:00+00:00"},
        top_k=5,
    )
    assert [item.record.id for item in results] == ["mem-recent"]


@pytest.mark.asyncio
async def test_access_roles_enforced_for_search_get_and_list():
    client = MemoryClient(InMemoryDriver())
    mem_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-2",
        content="Restricted travel note",
        source="reflection",
        access_roles=["support", "admin"],
    )

    with pytest.raises(PermissionError):
        await client.get(mem_id)

    record = await client.get(mem_id, access_roles=["support"])
    assert record.id == mem_id

    search_results = await client.search("restricted travel", filters={"user_id": "u-2"}, access_roles=["support"])
    assert [item.record.id for item in search_results] == [mem_id]

    listed = await client.list({"user_id": "u-2"}, access_roles=["support"])
    assert [item.id for item in listed] == [mem_id]


@pytest.mark.asyncio
async def test_run_with_attribution_returns_response_and_policy_trail():
    client = MemoryClient(
        config={
            "policies": {
                "extraction": [
                    {
                        "name": "extract-user-preferences",
                        "trigger": "conversation_turn",
                        "conditions": [{"content_matches": ["prefer"]}],
                        "create": {"type": "semantic", "scope": "user"},
                    }
                ]
            }
        },
        driver=InMemoryDriver(),
    )
    await client.add(
        type="semantic",
        scope="user",
        user_id="u-3",
        content="User prefers metric units.",
        source="conversation",
    )

    response, attribution = await client.run_with_attribution(
        "I prefer metric units.",
        user_id="u-3",
    )
    assert "Task:" in response
    assert "Relevant memories:" in response
    assert attribution.memories_used
    assert "extract-user-preferences" in attribution.policies_fired
