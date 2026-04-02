"""
Agent Attribution and Memory-Augmented Runners
===============================================

This example shows how to use Engram's attribution system to track
which memories influenced an LLM response and which policies fired.

Two approaches are demonstrated:

1. ``client.run_with_attribution()`` -- a lightweight method on the
   MemoryClient that accepts a custom ``runner`` callable.  No OpenAI
   dependency is needed when you supply your own runner.

2. ``EngramAgent`` -- a standalone agent class that wraps an LLM client
   and automatically enriches prompts with relevant memories.  The
   OpenAI-backed usage is shown in a commented section.

What this example covers:

- Creating a MemoryClient with pre-loaded memories
- Writing a custom runner (sync or async) for ``run_with_attribution``
- Inspecting the ``OutputAttribution`` object (memories_used, policies_fired)
- Using the ``EngramAgent`` class (commented -- needs ``openai`` package)

No external dependencies are needed.  Run directly:

    python examples/09_agent_attribution.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from engramx import MemoryClient
from engramx.config import EngramConfig
from engramx.observability import OutputAttribution, ScoredMemory


# =====================================================================
# Helper: a custom runner that needs no LLM
# =====================================================================

def mock_runner(prompt: str, memories: list[ScoredMemory]) -> str:
    """A simple runner that echoes the prompt and summarises the memories.

    ``run_with_attribution`` calls the runner with:
        runner(prompt: str, memories: list[ScoredMemory]) -> str | Awaitable[str]

    The runner receives the fully-assembled prompt (task + memory context +
    policy context) and the raw scored memories.  It must return a response
    string (or an awaitable that resolves to one).
    """
    memory_bullets = []
    for mem in memories:
        memory_bullets.append(
            f"  - (score={mem.score:.3f}) [{mem.record.type}] {mem.record.content}"
        )

    memory_section = "\n".join(memory_bullets) or "  (none)"

    return (
        f"[MockRunner] I received a prompt with {len(memories)} memory(ies).\n"
        f"\nMemories provided:\n{memory_section}\n"
        f"\nI would normally call an LLM here.  Instead, here is the raw prompt "
        f"(first 200 chars):\n  {prompt[:200]}..."
    )


# An async runner works too -- just return a coroutine
async def async_mock_runner(prompt: str, memories: list[ScoredMemory]) -> str:
    """Async variant of the mock runner."""
    return mock_runner(prompt, memories)


# =====================================================================
# 1. run_with_attribution -- custom runner, no OpenAI needed
# =====================================================================

async def attribution_demo() -> None:
    """Show the full run_with_attribution flow."""
    print("=" * 60)
    print("run_with_attribution demo (custom runner)")
    print("=" * 60)

    # -- Build a client with some policy rules so policies_fired is populated
    config = EngramConfig.from_mapping({
        "driver": {"kind": "memory"},
        "policies": {
            "extraction": [
                {
                    "name": "capture_conversation",
                    "trigger": "conversation_turn",
                    "create": {"type": "episodic", "scope": "session"},
                },
            ],
        },
    })
    client = MemoryClient(config=config)

    # -- Pre-load some memories ----------------------------------------
    await client.add(
        type="episodic",
        scope="user",
        content="User prefers dark mode in all applications.",
        user_id="alice",
        importance_score=0.9,
    )
    await client.add(
        type="semantic",
        scope="global",
        content="The company holiday party is on December 20th.",
        importance_score=0.7,
    )
    await client.add(
        type="procedural",
        scope="agent",
        content="Always greet the user by name when starting a conversation.",
        agent_id="assistant-v1",
        importance_score=0.8,
    )
    await client.add(
        type="episodic",
        scope="user",
        content="Alice asked about dark mode settings yesterday.",
        user_id="alice",
        importance_score=0.6,
    )

    print("\nLoaded 4 memories into the store.\n")

    # -- Run with a synchronous runner ---------------------------------
    print("-" * 40)
    print("Sync runner")
    print("-" * 40)

    response, attribution = await client.run_with_attribution(
        task="What are Alice's UI preferences?",
        user_id="alice",
        top_k=3,
        runner=mock_runner,
    )

    print(f"\nResponse:\n{response}\n")
    _print_attribution(attribution)

    # -- Run with an async runner --------------------------------------
    print("-" * 40)
    print("Async runner")
    print("-" * 40)

    response2, attribution2 = await client.run_with_attribution(
        task="Tell me about upcoming company events.",
        top_k=2,
        runner=async_mock_runner,
    )

    print(f"\nResponse:\n{response2}\n")
    _print_attribution(attribution2)

    # -- Run with NO runner (returns the raw prompt as the response) ---
    print("-" * 40)
    print("No runner (raw prompt returned)")
    print("-" * 40)

    raw_response, attribution3 = await client.run_with_attribution(
        task="How should I greet users?",
        top_k=3,
        runner=None,
    )

    print(f"\nRaw prompt (first 300 chars):\n  {raw_response[:300]}...\n")
    _print_attribution(attribution3)


def _print_attribution(attr: OutputAttribution) -> None:
    """Pretty-print an OutputAttribution object."""
    print("Attribution:")
    print(f"  memories_used  : {attr.memories_used}")
    print(f"  policies_fired : {attr.policies_fired}")
    print()


# =====================================================================
# 2. build_attribution -- manual construction
# =====================================================================

async def manual_attribution_demo() -> None:
    """Show how to build an OutputAttribution object manually."""
    print("=" * 60)
    print("Manual build_attribution demo")
    print("=" * 60)

    client = MemoryClient(driver="memory")

    mem_id_1 = await client.add(
        type="semantic",
        scope="global",
        content="Python was created by Guido van Rossum.",
    )
    mem_id_2 = await client.add(
        type="semantic",
        scope="global",
        content="The Zen of Python has 19 aphorisms.",
    )

    # Search to get ScoredMemory objects
    results = await client.search("Python creator", top_k=2)

    # Build attribution from the search results
    attribution = client.build_attribution(
        memories=results,
        policies_fired=["capture_conversation", "decay_episodic"],
    )

    print(f"\nSearch returned {len(results)} result(s):")
    for item in results:
        print(f"  [{item.record.type}] {item.record.content} (score={item.score:.3f})")

    print()
    _print_attribution(attribution)


# =====================================================================
# 3. EngramAgent (commented -- requires the openai package)
# =====================================================================

# The EngramAgent class provides a higher-level interface that
# automatically searches memories, builds a context prompt, calls the
# LLM, and returns the response with full attribution.
#
# from engramx.agent import EngramAgent
#
# async def agent_demo() -> None:
#     client = MemoryClient(driver="memory")
#
#     # Pre-load some memories
#     await client.add(
#         type="semantic",
#         scope="global",
#         content="The user's preferred language is Python.",
#     )
#     await client.add(
#         type="episodic",
#         scope="user",
#         content="User asked how to read a CSV file yesterday.",
#         user_id="alice",
#     )
#
#     # --- Default: uses openai.AsyncOpenAI() -------------------------
#     agent = EngramAgent(client=client, model="gpt-4o-mini")
#
#     response, attribution = await agent.run_with_attribution(
#         task="How do I read a CSV in the user's preferred language?",
#         user_id="alice",
#         top_k=3,
#     )
#
#     print(f"Agent response:\n{response}\n")
#     print(f"Memories used : {attribution.memories_used}")
#     print(f"Policies fired: {attribution.policies_fired}")
#
#     # --- Custom LLM client (e.g. Azure, local endpoint) -------------
#     # import openai
#     # custom_llm = openai.AsyncOpenAI(
#     #     base_url="https://my-azure.openai.azure.com/",
#     #     api_key="...",
#     # )
#     # agent = EngramAgent(client=client, llm_client=custom_llm, model="gpt-4o")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    asyncio.run(attribution_demo())
    asyncio.run(manual_attribution_demo())
