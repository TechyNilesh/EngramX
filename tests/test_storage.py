import pytest

from engramxx.client import MemoryClient
from engramxx.schema import MemoryRecord
from engramxx.storage.memory import InMemoryDriver
from engramxx.storage.sqlite import SQLiteDriver


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_factory", [InMemoryDriver, lambda: SQLiteDriver(":memory:")])
async def test_crud_search_and_timeline(driver_factory):
    client = MemoryClient(driver_factory())

    mem_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User prefers metric units and kilograms.",
        source="conversation",
        importance_score=0.9,
        sensitivity_level="internal",
    )

    stored = await client.get(mem_id)
    assert stored.user_id == "u-123"
    assert stored.access_count == 1

    updated = await client.update(mem_id, {"content": "User prefers metric units and Celsius.", "importance_score": 1.0})
    assert updated.content.endswith("Celsius.")
    assert updated.importance_score == 1.0

    results = await client.search("metric temperature preferences", filters={"scope": "user", "user_id": "u-123"}, top_k=5)
    assert results
    assert results[0].record.id == mem_id
    assert "metric" in results[0].matched_terms

    trace = await client.explain("metric temperature preferences", filters={"scope": "user", "user_id": "u-123"})
    assert trace.retrieved
    assert mem_id in trace.to_markdown()

    timeline = await client.timeline(user_id="u-123")
    assert [event.operation for event in timeline] == ["upsert", "get", "update"]


@pytest.mark.asyncio
async def test_delete_by_filters():
    client = MemoryClient(InMemoryDriver())
    keep_id = await client.add(type="semantic", scope="user", user_id="u-1", content="keep", source="conversation")
    delete_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-2",
        content="delete",
        source="conversation",
        sensitivity_level="confidential",
    )

    deleted = await client.delete({"user_id": "u-2", "sensitivity_level": "confidential"})
    assert deleted == 1
    remaining = await client.search("keep", filters={"scope": "user"}, top_k=10)
    assert [item.record.id for item in remaining] == [keep_id]

    with pytest.raises(KeyError):
        await client.get(delete_id)

