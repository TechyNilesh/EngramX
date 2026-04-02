"""
Framework Adapters Example
===========================

Shows how to use Engram's built-in adapters for three popular AI frameworks:

1. **LangChain** -- ``EngramChatMemory``
   load_memory_variables, save_context, clear

2. **LlamaIndex** -- ``EngramMemoryBlock``
   get, put, get_all, reset

3. **AutoGen** -- ``EngramAutoGenMemory``
   query, add, update_context

All adapters are backed by the in-memory driver so this example works
without installing LangChain, LlamaIndex, or AutoGen.

Run this file directly:

    python examples/07_framework_adapters.py
"""

from __future__ import annotations

from engramx import MemoryClient
from engramx.adapters.langchain import EngramChatMemory
from engramx.adapters.llamaindex import EngramMemoryBlock
from engramx.adapters.autogen import EngramAutoGenMemory


def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. LangChain adapter -- EngramChatMemory
# ---------------------------------------------------------------------------

def langchain_demo() -> None:
    separator("LangChain Adapter: EngramChatMemory")

    client = MemoryClient(driver="memory")
    memory = EngramChatMemory(client=client, user_id="alice", top_k=3)

    # -- memory_variables -----------------------------------------------------
    print(f"\nmemory_variables: {memory.memory_variables}")

    # -- save_context (store two conversation turns) --------------------------
    memory.save_context(
        {"input": "What is the capital of France?"},
        {"output": "The capital of France is Paris."},
    )
    print("Saved context: 'What is the capital of France?' -> 'Paris'")

    memory.save_context(
        {"input": "Tell me about the Eiffel Tower."},
        {"output": "The Eiffel Tower is a famous landmark in Paris, built in 1889."},
    )
    print("Saved context: 'Tell me about the Eiffel Tower.' -> 'landmark in Paris'")

    memory.save_context(
        {"input": "What language do they speak there?"},
        {"output": "French is the official language of France."},
    )
    print("Saved context: 'What language do they speak there?' -> 'French'")

    # -- load_memory_variables (search for relevant memories) -----------------
    result = memory.load_memory_variables({"input": "Paris landmarks"})
    print(f"\nload_memory_variables('Paris landmarks'):")
    for line in result["history"].splitlines():
        print(f"  {line}")

    result2 = memory.load_memory_variables({"input": "language"})
    print(f"\nload_memory_variables('language'):")
    for line in result2["history"].splitlines():
        print(f"  {line}")

    # -- clear ----------------------------------------------------------------
    memory.clear()
    after_clear = memory.load_memory_variables({"input": "anything"})
    print(f"\nAfter clear(), history is empty: {after_clear['history'] == ''}")


# ---------------------------------------------------------------------------
# 2. LlamaIndex adapter -- EngramMemoryBlock
# ---------------------------------------------------------------------------

def llamaindex_demo() -> None:
    separator("LlamaIndex Adapter: EngramMemoryBlock")

    client = MemoryClient(driver="memory")
    block = EngramMemoryBlock(client=client, user_id="bob", top_k=3)

    # -- put (store several pieces of knowledge) ------------------------------
    id1 = block.put("Engram supports episodic, semantic, and procedural memory types.")
    print(f"\nput() -> stored memory {id1[:12]}...")

    id2 = block.put("The in-memory driver is great for testing and prototyping.")
    print(f"put() -> stored memory {id2[:12]}...")

    id3 = block.put("LlamaIndex agents can use EngramMemoryBlock for long-term recall.")
    print(f"put() -> stored memory {id3[:12]}...")

    # -- get (search for relevant memories) -----------------------------------
    result = block.get("memory types")
    print(f"\nget('memory types'):")
    for line in result.splitlines():
        print(f"  {line}")

    result2 = block.get("testing")
    print(f"\nget('testing'):")
    for line in result2.splitlines():
        print(f"  {line}")

    # -- get_all (list every stored memory) -----------------------------------
    all_memories = block.get_all()
    print(f"\nget_all() returned {len(all_memories)} memories:")
    for mem in all_memories:
        print(f"  [{mem['type']}] {mem['content'][:70]}...")

    # -- reset (clear all memories for this user) -----------------------------
    block.reset()
    after_reset = block.get_all()
    print(f"\nAfter reset(), memories remaining: {len(after_reset)}")


# ---------------------------------------------------------------------------
# 3. AutoGen adapter -- EngramAutoGenMemory
# ---------------------------------------------------------------------------

def autogen_demo() -> None:
    separator("AutoGen Adapter: EngramAutoGenMemory")

    client = MemoryClient(driver="memory")
    memory = EngramAutoGenMemory(client=client, user_id="charlie", top_k=3)

    # -- add (store memories) -------------------------------------------------
    id1 = memory.add("Charlie prefers concise answers.")
    print(f"\nadd() -> stored memory {id1[:12]}...")

    id2 = memory.add("Charlie is working on a machine learning project.")
    print(f"add() -> stored memory {id2[:12]}...")

    id3 = memory.add("The project deadline is next Friday.")
    print(f"add() -> stored memory {id3[:12]}...")

    # -- query (search for relevant memories) ---------------------------------
    results = memory.query("project deadline")
    print(f"\nquery('project deadline') returned {len(results)} result(s):")
    for item in results:
        print(f"  [{item['type']}] {item['content']} (score={item['score']:.3f})")

    results2 = memory.query("answer preferences")
    print(f"\nquery('answer preferences') returned {len(results2)} result(s):")
    for item in results2:
        print(f"  [{item['type']}] {item['content']} (score={item['score']:.3f})")

    # -- update_context (inject memories into a message list) -----------------
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "When is my project due?"},
    ]

    print("\nMessages BEFORE update_context:")
    for msg in messages:
        print(f"  [{msg['role']}] {msg['content'][:70]}")

    augmented = memory.update_context(agent=None, messages=messages)

    print(f"\nMessages AFTER update_context ({len(augmented)} total):")
    for msg in augmented:
        content_preview = msg["content"][:70]
        if len(msg["content"]) > 70:
            content_preview += "..."
        print(f"  [{msg['role']}] {content_preview}")

    # -- clear ----------------------------------------------------------------
    deleted = memory.clear()
    print(f"\nclear() deleted {deleted} memories")
    remaining = memory.query("anything")
    print(f"After clear(), query returns {len(remaining)} results")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    langchain_demo()
    llamaindex_demo()
    autogen_demo()

    print("\n" + "=" * 60)
    print("  All adapter demos completed successfully!")
    print("=" * 60)
