"""
Engram Observability Example
=============================

This example demonstrates the observability features built into Engram:

1. Creating a MemoryClient with the in-memory driver
2. Adding memories with varying types, importance scores, and decay factors
3. Using ``explain()`` to get a retrieval trace with full scoring details
4. Using ``timeline()`` to inspect ordered memory events
5. Using ``build_attribution()`` to track which memories influenced an output
6. Using ``run_with_attribution()`` with a custom runner function

These tools help you understand *why* certain memories were retrieved, *when*
memory operations occurred, and *which* memories contributed to a given output.

No external dependencies are needed -- the in-memory driver ships with Engram.
Run this file directly:

    python examples/03_observability.py
"""

import asyncio

from engramx import MemoryClient


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Create a client with the in-memory driver
    # ------------------------------------------------------------------

    client = MemoryClient(driver="memory")

    # ------------------------------------------------------------------
    # 2. Add several memories with different types, importance, and decay
    # ------------------------------------------------------------------

    await client.add(
        type="episodic",
        scope="user",
        content="User prefers dark mode across all applications.",
        user_id="user-42",
        importance_score=0.9,
        decay_factor=1.0,
    )

    await client.add(
        type="semantic",
        scope="global",
        content="Dark mode reduces eye strain in low-light environments.",
        importance_score=0.7,
        decay_factor=0.8,
    )

    await client.add(
        type="procedural",
        scope="agent",
        content="When generating UI code, always check the user's theme preference first.",
        agent_id="assistant-v1",
        importance_score=0.6,
        decay_factor=0.95,
    )

    await client.add(
        type="episodic",
        scope="user",
        content="User reported a bug in the notification settings panel.",
        user_id="user-42",
        importance_score=0.4,
        decay_factor=0.5,
    )

    await client.add(
        type="semantic",
        scope="global",
        content="CSS media query prefers-color-scheme detects OS-level dark mode.",
        importance_score=0.8,
        decay_factor=0.9,
    )

    print("Added 5 memories with varying types, importance, and decay.\n")

    # ------------------------------------------------------------------
    # 3. explain() -- get a full retrieval trace
    # ------------------------------------------------------------------

    print("=" * 60)
    print("EXPLAIN: Retrieval trace for 'dark mode preference'")
    print("=" * 60)

    trace = await client.explain("dark mode preference", top_k=3)

    # Print the built-in markdown report
    print(trace.to_markdown())
    print()

    # You can also inspect individual retrieved entries programmatically
    for entry in trace.retrieved:
        print(
            f"  Memory {entry.id[:12]}...  "
            f"score={entry.score:.3f}  "
            f"decay_adjusted={entry.decay_adjusted:.3f}  "
            f"terms={entry.matched_terms}"
        )

    # The trace also exposes the full candidate list (before top_k truncation)
    print(f"\nTotal candidates evaluated: {len(trace.candidates)}")
    print(f"Top-k returned: {len(trace.retrieved)}")

    # ------------------------------------------------------------------
    # 4. timeline() -- ordered memory events
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("TIMELINE: All memory events in chronological order")
    print("=" * 60)

    events = await client.timeline()

    for event in events:
        print(
            f"  {event.operation:<8s}  "
            f"memory_id={event.memory_id[:12]}...  "
            f"timestamp={event.timestamp.isoformat()}"
        )

    # You can also filter the timeline by user_id
    print("\nTimeline filtered to user-42:")
    user_events = await client.timeline(user_id="user-42")
    for event in user_events:
        print(
            f"  {event.operation:<8s}  "
            f"memory_id={event.memory_id[:12]}...  "
            f"timestamp={event.timestamp.isoformat()}"
        )

    # ------------------------------------------------------------------
    # 5. build_attribution() -- track which memories influenced output
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("ATTRIBUTION: Which memories influenced our response?")
    print("=" * 60)

    # First, run a search to get scored memories
    results = await client.search("dark mode theme", top_k=3)

    # Build an attribution object from the search results
    attribution = client.build_attribution(
        memories=results,
        policies_fired=["theme_preference_check", "ui_guideline_lookup"],
    )

    print(f"Memories used ({len(attribution.memories_used)}):")
    for mid in attribution.memories_used:
        print(f"  - {mid}")

    print(f"Policies fired ({len(attribution.policies_fired)}):")
    for policy in attribution.policies_fired:
        print(f"  - {policy}")

    # ------------------------------------------------------------------
    # 6. run_with_attribution() -- end-to-end task with attribution
    # ------------------------------------------------------------------

    print()
    print("=" * 60)
    print("RUN WITH ATTRIBUTION: Execute a task with memory context")
    print("=" * 60)

    # Define a simple runner that formats memory context into a response.
    # In practice this would call an LLM or another processing pipeline.
    def my_runner(prompt: str, memories: list) -> str:
        lines = [f"[Runner received {len(memories)} memories]"]
        for mem in memories:
            lines.append(f"  - {mem.record.content[:60]}")
        lines.append(f"[Prompt starts with: {prompt[:80]}...]")
        return "\n".join(lines)

    response, attr = await client.run_with_attribution(
        task="What theme should I use for the new dashboard?",
        user_id="user-42",
        top_k=3,
        runner=my_runner,
    )

    print("Response from runner:")
    print(response)
    print()
    print(f"Attribution -- memories used: {len(attr.memories_used)}")
    for mid in attr.memories_used:
        print(f"  - {mid}")
    print(f"Attribution -- policies fired: {attr.policies_fired}")


if __name__ == "__main__":
    asyncio.run(main())
