"""
Chat with Memory Example
=========================

Demonstrates how to use Engram as persistent memory for a conversational AI.
A mock LLM (no OpenAI or other external service required) carries on a
multi-turn conversation while Engram stores and retrieves relevant memories.

Key concepts shown:

1. Creating a MemoryClient with the in-memory driver
2. Simulating a multi-turn conversation with a simple mock LLM
3. Storing each turn as episodic memory via ``add_sync``
4. Retrieving relevant memories before generating each response via ``search_sync``
5. Watching memories accumulate over turns

Run this file directly:

    python examples/06_chat_with_memory.py
"""

from __future__ import annotations

from engramx import MemoryClient


# ---------------------------------------------------------------------------
# Mock LLM -- no external dependencies needed
# ---------------------------------------------------------------------------

# A handful of canned responses keyed by simple keyword matches.  When no
# keyword matches the mock LLM echoes the user message back.

_CANNED_RESPONSES: list[tuple[list[str], str]] = [
    (
        ["hello", "hi", "hey"],
        "Hello! How can I help you today?",
    ),
    (
        ["python", "programming", "code"],
        "Python is a fantastic language! I especially enjoy its clean syntax.",
    ),
    (
        ["weather", "rain", "sunny"],
        "I don't have live weather data, but I hope it's nice where you are!",
    ),
    (
        ["food", "cook", "recipe", "pizza"],
        "I love talking about food! A simple margherita pizza is hard to beat.",
    ),
    (
        ["book", "read", "reading"],
        "Reading is wonderful. Have you tried any good non-fiction lately?",
    ),
]


def mock_llm(user_message: str, memory_context: str) -> str:
    """Generate a response using keyword matching and memory context.

    If relevant memories exist the mock LLM weaves them into its reply so you
    can see how recall affects the conversation.
    """
    lower = user_message.lower()

    # Pick the first canned response whose keywords overlap the user message.
    base_response: str | None = None
    for keywords, response in _CANNED_RESPONSES:
        if any(kw in lower for kw in keywords):
            base_response = response
            break

    if base_response is None:
        base_response = f"Interesting -- you said: '{user_message}'"

    # If we have recalled memories, append a note about what the LLM remembers.
    if memory_context:
        return (
            f"{base_response}\n"
            f"  (I also recall from our conversation: {memory_context})"
        )
    return base_response


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat_loop() -> None:
    """Run a simulated multi-turn conversation with memory."""

    # 1. Create an in-memory client -- nothing written to disk.
    client = MemoryClient(driver="memory")

    user_id = "demo-user"

    # Pre-scripted turns so the example is fully self-contained.
    turns = [
        "Hi there!",
        "I've been learning Python lately.",
        "Do you have a good pizza recipe?",
        "What were we talking about earlier?",
        "I also enjoy reading books about programming.",
    ]

    print("=" * 60)
    print("  Engram Chat-with-Memory Demo")
    print("=" * 60)

    for turn_number, user_message in enumerate(turns, start=1):
        print(f"\n--- Turn {turn_number} ---")
        print(f"User: {user_message}")

        # 4. Search for memories relevant to the current message.
        recalled = client.search_sync(user_message, top_k=3)

        memory_context = ""
        if recalled:
            memory_lines = [item.record.content for item in recalled]
            memory_context = " | ".join(memory_lines)
            print(f"  [memories recalled: {len(recalled)}]")
            for item in recalled:
                print(f"    - {item.record.content}")
        else:
            print("  [no memories recalled yet]")

        # 2. Generate a response using the mock LLM.
        assistant_response = mock_llm(user_message, memory_context)
        print(f"Assistant: {assistant_response}")

        # 3. Store this turn as an episodic memory.
        content = f"User said: '{user_message}' -- Assistant replied: '{assistant_response.split(chr(10))[0]}'"
        mem_id = client.add_sync(
            type="episodic",
            scope="user",
            user_id=user_id,
            content=content,
            source="chat",
        )
        print(f"  [stored as memory {mem_id[:12]}...]")

    # 5. Show how memories have accumulated.
    print("\n" + "=" * 60)
    print("  All memories after the conversation")
    print("=" * 60)

    all_results = client.search_sync("conversation summary", top_k=10)
    for i, item in enumerate(all_results, start=1):
        print(f"  {i}. [{item.record.type}] {item.record.content[:90]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chat_loop()
