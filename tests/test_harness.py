import pytest

from engramxx.testing import MemoryHarness


@pytest.mark.asyncio
async def test_memory_harness_searches_conversation():
    harness = MemoryHarness(driver="memory")
    await harness.inject_conversation(
        [
            ("user", "I always prefer metric units."),
            ("assistant", "Noted."),
        ]
    )
    await harness.run_policies()
    results = await harness.search("unit preferences", type="episodic")
    assert results
    assert "unit" in results[0].matched_terms or "preferences" in results[0].matched_terms

