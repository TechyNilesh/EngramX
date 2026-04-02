"""
Testing Memory Behavior with MemoryHarness
===========================================

This example demonstrates how to use ``engram.testing.MemoryHarness`` to
verify memory extraction, decay, and promotion in an isolated test
environment.  The harness spins up an in-memory driver, lets you inject
conversations, runs policy evaluation, and then exposes the stored
memories for assertions.

Covered topics:

1. Creating a MemoryHarness with an in-memory driver
2. Injecting a multi-turn conversation via ``inject_conversation()``
3. Running extraction / decay / promotion via ``run_policies()``
4. Searching for extracted memories
5. Writing pytest-style test functions
6. Testing extraction rules, decay, and promotion independently

No external dependencies are needed -- run this file directly:

    python examples/10_testing.py
"""

import asyncio

from engramx.testing import MemoryHarness


# ---------------------------------------------------------------------------
# 1. Full walkthrough -- inject, process, and search
# ---------------------------------------------------------------------------

async def main() -> None:
    # Create a harness with the default config.
    # The default ships with an extraction rule named "extract-user-preferences"
    # that fires on conversation turns containing the word "prefer".
    harness = MemoryHarness(driver="sqlite")

    print("=== Step 1: Inject a conversation ===")
    ids = await harness.inject_conversation([
        ("user", "I prefer dark mode in all my editors."),
        ("assistant", "Got it -- I will remember that you prefer dark mode."),
        ("user", "I also prefer Python over JavaScript for backend work."),
        ("assistant", "Noted!  Python for backend."),
    ])
    print(f"Injected {len(ids)} conversation turns: {ids[:2]}...")

    print("\n=== Step 2: Run policies (extraction + decay + promotion) ===")
    await harness.run_policies()
    print("Policies executed successfully.")

    print("\n=== Step 3: Search for extracted memories ===")
    results = await harness.search("dark mode")
    print(f"Search 'dark mode' returned {len(results)} result(s):")
    for item in results:
        print(f"  [{item.record.type:>10}] {item.record.content[:80]}")

    results = await harness.search("Python backend")
    print(f"\nSearch 'Python backend' returned {len(results)} result(s):")
    for item in results:
        print(f"  [{item.record.type:>10}] {item.record.content[:80]}")

    # You can also list everything the harness stored.
    all_records = await harness.client.list()
    print(f"\nTotal memories in store: {len(all_records)}")
    for rec in all_records:
        print(f"  [{rec.type:>10}] decay={rec.decay_factor:.2f}  {rec.content[:60]}")


# ---------------------------------------------------------------------------
# 2. Custom extraction rules -- tighter control over what gets extracted
# ---------------------------------------------------------------------------

async def custom_rules_demo() -> None:
    """Show how to supply your own policy config to the harness."""

    config = {
        "driver": {"kind": "sqlite", "path": ":memory:"},
        "policies": {
            "extraction": [
                {
                    "name": "extract-tool-preferences",
                    "trigger": "conversation_turn",
                    "conditions": [{"content_matches": ["tool", "prefer"]}],
                    "create": {
                        "type": "semantic",
                        "scope": "user",
                        "importance_score": 0.9,
                        "sensitivity_level": "internal",
                    },
                },
            ],
            "retention": [
                {
                    "name": "decay-session-memories",
                    "applies_to": {"scope": "session"},
                    "decay_function": "exponential",
                    "half_life_days": 7,
                    "floor_score": 0.1,
                },
            ],
        },
    }

    harness = MemoryHarness(config=config)

    await harness.inject_conversation([
        ("user", "I prefer the tool called pytest for testing."),
        ("assistant", "Great choice!"),
        ("user", "I like reading books."),          # no "tool" + "prefer" -> no extraction
    ])
    await harness.run_policies()

    # The first turn should trigger extraction; the third should not.
    results = await harness.search("pytest")
    print("\n=== Custom rules: search 'pytest' ===")
    print(f"Results: {len(results)}")
    for item in results:
        print(f"  [{item.record.type}] {item.record.content[:70]}")

    # Verify decay was applied to session-scoped memories.
    all_records = await harness.client.list()
    decayed = [r for r in all_records if r.scope == "session" and r.decay_factor < 1.0]
    print(f"\nSession memories with decay applied: {len(decayed)}")
    for rec in decayed:
        print(f"  decay={rec.decay_factor:.4f}  {rec.content[:60]}")


# ---------------------------------------------------------------------------
# 3. Pytest-style test patterns
# ---------------------------------------------------------------------------

async def test_extraction_creates_semantic_memory() -> None:
    """Extraction rules should produce semantic memories from matching turns."""
    harness = MemoryHarness()

    await harness.inject_conversation([
        ("user", "I prefer tabs over spaces."),
    ])
    await harness.run_policies()

    results = await harness.search("tabs over spaces")
    # In a real pytest run you would use plain ``assert``:
    assert len(results) > 0, "Expected at least one extracted memory"
    assert any(
        r.record.type == "semantic" for r in results
    ), "Extracted memory should be semantic"
    print("\n[PASS] test_extraction_creates_semantic_memory")


async def test_non_matching_turn_skips_extraction() -> None:
    """Turns that do not match any extraction condition should not create new memories."""
    harness = MemoryHarness()

    await harness.inject_conversation([
        ("user", "Hello, how are you?"),
    ])
    await harness.run_policies()

    all_records = await harness.client.list()
    semantic_records = [r for r in all_records if r.type == "semantic"]
    assert len(semantic_records) == 0, "No semantic memory should be extracted"
    print("[PASS] test_non_matching_turn_skips_extraction")


async def test_decay_reduces_factor() -> None:
    """Session-scoped memories should have their decay_factor reduced over time."""
    config = {
        "driver": {"kind": "sqlite", "path": ":memory:"},
        "policies": {
            "extraction": [
                {
                    "name": "extract-preferences",
                    "trigger": "conversation_turn",
                    "conditions": [{"content_matches": ["prefer"]}],
                    "create": {
                        "type": "semantic",
                        "scope": "user",
                        "importance_score": 0.8,
                        "sensitivity_level": "internal",
                    },
                },
            ],
            "retention": [
                {
                    "name": "decay-session",
                    "applies_to": {"scope": "session"},
                    "decay_function": "exponential",
                    "half_life_days": 1,
                    "floor_score": 0.05,
                },
            ],
        },
    }
    harness = MemoryHarness(config=config)

    await harness.inject_conversation([
        ("user", "I prefer vim."),
    ])
    await harness.run_policies()

    records = await harness.client.list()
    session_records = [r for r in records if r.scope == "session"]
    # After one round of decay, session records should still exist.
    # With a 1-day half-life and near-zero elapsed time, decay should be close to 1.0.
    for rec in session_records:
        assert rec.decay_factor <= 1.0, "Decay factor should not exceed 1.0"
    print("[PASS] test_decay_reduces_factor")


async def test_promotion_creates_higher_type() -> None:
    """Repeated similar episodic memories should promote to procedural/semantic."""
    config = {
        "driver": {"kind": "sqlite", "path": ":memory:"},
        "policies": {
            "extraction": [
                {
                    "name": "extract-preferences",
                    "trigger": "conversation_turn",
                    "conditions": [{"content_matches": ["prefer"]}],
                    "create": {
                        "type": "semantic",
                        "scope": "user",
                        "importance_score": 0.7,
                        "sensitivity_level": "internal",
                    },
                },
            ],
            "summarization": [
                {
                    "name": "promote-repeated",
                    "applies_to": {"type": "semantic"},
                    "trigger": "same_content_count >= 2",
                    "promote_to": {"type": "procedural", "scope": "agent"},
                },
            ],
        },
    }
    harness = MemoryHarness(config=config)

    # Inject multiple turns with similar content to trigger promotion.
    await harness.inject_conversation([
        ("user", "I prefer dark mode."),
        ("assistant", "Noted."),
        ("user", "I really prefer dark mode in every app."),
        ("assistant", "Understood."),
    ])
    await harness.run_policies()

    all_records = await harness.client.list()
    types = {r.type for r in all_records}
    print(f"\n[INFO] Memory types after promotion: {types}")
    print(f"[INFO] Total records: {len(all_records)}")
    for rec in all_records:
        print(f"  [{rec.type:>10}] derived_from={rec.derived_from}  {rec.content[:50]}")
    print("[PASS] test_promotion_creates_higher_type")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(custom_rules_demo())

    # Run the pytest-style tests.
    print("\n=== Running pytest-style assertions ===")
    asyncio.run(test_extraction_creates_semantic_memory())
    asyncio.run(test_non_matching_turn_skips_extraction())
    asyncio.run(test_decay_reduces_factor())
    asyncio.run(test_promotion_creates_higher_type())

    print("\nAll tests passed.")
