"""
Engram Quickstart Example
=========================

This example walks through the core operations of the Engram memory framework:

1. Creating a MemoryClient with the built-in in-memory driver
2. Adding episodic, semantic, and procedural memories
3. Searching memories with filters
4. Retrieving a single memory by ID
5. Updating a memory
6. Deleting memories by filter
7. Using synchronous wrappers (add_sync, search_sync)

No external dependencies are needed -- the in-memory driver ships with Engram.
Run this file directly:

    python examples/01_quickstart.py
"""

import asyncio

from engramx import MemoryClient


# ---------------------------------------------------------------------------
# Async workflow -- covers steps 1-6
# ---------------------------------------------------------------------------

async def main() -> None:
    # 1. Create a client backed by the in-memory driver.
    #    Nothing is persisted to disk; perfect for experiments.
    client = MemoryClient(driver="memory")

    # ------------------------------------------------------------------
    # 2. Add memories of different types
    # ------------------------------------------------------------------

    # Episodic memory -- a specific event or experience
    episodic_id = await client.add(
        type="episodic",
        scope="user",
        content="Had a great meeting with the design team about the new dashboard.",
        user_id="alice",
        importance_score=0.8,
    )
    print(f"Added episodic memory: {episodic_id}")

    # Semantic memory -- a general fact or piece of knowledge
    semantic_id = await client.add(
        type="semantic",
        scope="global",
        content="Python 3.12 introduced type parameter syntax for generic classes.",
        source="documentation",
        importance_score=0.9,
    )
    print(f"Added semantic memory: {semantic_id}")

    # Procedural memory -- a learned rule or how-to
    procedural_id = await client.add(
        type="procedural",
        scope="agent",
        content="When the user asks for a summary, retrieve the last 5 episodic memories first.",
        agent_id="assistant-v1",
        importance_score=0.7,
    )
    print(f"Added procedural memory: {procedural_id}")

    # Add one more episodic memory so searches return multiple results
    await client.add(
        type="episodic",
        scope="user",
        content="Reviewed pull request #42 for the authentication module.",
        user_id="alice",
        importance_score=0.6,
    )

    # ------------------------------------------------------------------
    # 3. Search memories with filters
    # ------------------------------------------------------------------

    # Search across all memories
    results = await client.search("dashboard design", top_k=3)
    print(f"\nSearch for 'dashboard design' -- {len(results)} result(s):")
    for item in results:
        print(f"  [{item.record.type}] {item.record.content}")

    # Search with a filter to narrow down to episodic memories only
    filtered = await client.search(
        "meeting",
        filters={"type": "episodic"},
        top_k=3,
    )
    print(f"\nFiltered search (episodic only) -- {len(filtered)} result(s):")
    for item in filtered:
        print(f"  [{item.record.type}] {item.record.content}")

    # ------------------------------------------------------------------
    # 4. Get a single memory by its ID
    # ------------------------------------------------------------------

    record = await client.get(semantic_id)
    print(f"\nRetrieved memory {record.id}:")
    print(f"  type    = {record.type}")
    print(f"  content = {record.content}")
    print(f"  source  = {record.source}")

    # ------------------------------------------------------------------
    # 5. Update a memory
    # ------------------------------------------------------------------

    updated = await client.update(semantic_id, {"importance_score": 1.0})
    print(f"\nUpdated importance_score: {updated.importance_score}")

    # ------------------------------------------------------------------
    # 6. Delete memories by filter
    # ------------------------------------------------------------------

    deleted_count = await client.delete({"type": "procedural"})
    print(f"\nDeleted {deleted_count} procedural memory(ies)")

    # Verify it is gone by listing what remains
    remaining = await client.list()
    print(f"Remaining memories: {len(remaining)}")
    for rec in remaining:
        print(f"  [{rec.type}] {rec.content[:60]}")


# ---------------------------------------------------------------------------
# Sync wrappers -- step 7
# ---------------------------------------------------------------------------

def sync_demo() -> None:
    """Demonstrate the synchronous convenience methods.

    ``add_sync`` and ``search_sync`` let you use Engram without writing
    async code -- handy for scripts, notebooks, or quick prototyping.
    """
    client = MemoryClient(driver="memory")

    # add_sync returns the memory ID, just like the async version
    mem_id = client.add_sync(
        type="semantic",
        scope="global",
        content="The sync wrappers call asyncio.run() under the hood.",
    )
    print(f"\n[sync] Added memory: {mem_id}")

    # search_sync returns a list of ScoredMemory objects
    results = client.search_sync("sync wrappers", top_k=3)
    print(f"[sync] Search returned {len(results)} result(s):")
    for item in results:
        print(f"  [{item.record.type}] {item.record.content}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run the async workflow
    asyncio.run(main())

    # Run the sync demo (outside of asyncio.run so there is no active loop)
    sync_demo()
