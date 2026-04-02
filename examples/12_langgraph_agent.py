"""
Engram + LangGraph: Long-term memory for stateful agents.

This example shows how to use Engram as the long-term memory layer in a
LangGraph agent.  LangGraph manages short-term state (checkpoints) while
Engram handles persistent, cross-session memory with search, decay, and
policy-driven lifecycle.

Architecture:
    START → [recall_memories] → [call_llm] → [persist_memories] → END
                  │                                │
                  ▼                                ▼
            engram.search()                  engram.add()

Requirements (for the full LangGraph version):
    pip install engram langgraph langchain-openai

The first half of this file runs WITHOUT any external deps (mock LangGraph
pattern using plain async functions).  The second half shows the real
LangGraph integration (commented, needs langgraph + langchain-openai).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from engramx import MemoryClient


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — Runnable demo (no external deps)
# ═══════════════════════════════════════════════════════════════════════
# This simulates the LangGraph node pattern using plain async functions
# and a mock LLM, so you can run it immediately.


@dataclass
class Message:
    role: str
    content: str


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    memory_context: str = ""
    user_id: str = "default"


async def recall_memories(state: AgentState, client: MemoryClient) -> None:
    """Node 1: Retrieve relevant Engram memories before LLM call."""
    user_msg = state.messages[-1].content if state.messages else ""

    memories = await client.search(
        query=user_msg,
        filters={"user_id": state.user_id},
        top_k=5,
    )

    if memories:
        state.memory_context = "\n".join(
            f"  - {m.record.content}" for m in memories
        )
    else:
        state.memory_context = "  (no relevant memories)"


def mock_llm(system_prompt: str, user_message: str) -> str:
    """Mock LLM that echoes back what it knows from memory."""
    if "metric" in system_prompt.lower():
        return "Based on your preferences, I'll use metric units (km, kg, Celsius)."
    if "tokyo" in user_message.lower():
        return "I'll search for flights to Tokyo. Let me check visa requirements first."
    if "allergic" in system_prompt.lower() or "peanut" in system_prompt.lower():
        return "I'll make sure to avoid peanuts in all meal recommendations."
    return f"I'll help you with: {user_message}"


async def call_llm(state: AgentState) -> None:
    """Node 2: Call LLM with memory-augmented context."""
    system_prompt = (
        "You are a helpful assistant with memory.\n"
        f"Relevant memories:\n{state.memory_context}"
    )
    user_msg = state.messages[-1].content

    response = mock_llm(system_prompt, user_msg)
    state.messages.append(Message(role="assistant", content=response))


async def persist_memories(state: AgentState, client: MemoryClient) -> None:
    """Node 3: Store the conversation turn as episodic memory."""
    if len(state.messages) >= 2:
        user_msg = state.messages[-2].content
        ai_msg = state.messages[-1].content
        await client.add(
            type="episodic",
            scope="user",
            user_id=state.user_id,
            content=f"User: {user_msg}\nAssistant: {ai_msg}",
            source="conversation",
            importance_score=0.7,
            sensitivity_level="internal",
        )


async def run_agent_turn(
    state: AgentState,
    client: MemoryClient,
    user_message: str,
) -> str:
    """Execute one agent turn: recall → llm → persist."""
    state.messages.append(Message(role="user", content=user_message))

    # Node 1: Recall
    await recall_memories(state, client)

    # Node 2: LLM
    await call_llm(state)

    # Node 3: Persist
    await persist_memories(state, client)

    return state.messages[-1].content


async def demo_mock_langgraph() -> None:
    """Run a multi-turn conversation using the mock LangGraph pattern."""
    print("=" * 60)
    print("PART 1: Mock LangGraph pattern (no external deps)")
    print("=" * 60)

    client = MemoryClient(driver="memory")

    # Pre-load some long-term memories
    await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User prefers metric units for all measurements.",
        source="conversation",
        importance_score=0.9,
    )
    await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User is allergic to peanuts.",
        source="conversation",
        importance_score=0.95,
    )
    await client.add(
        type="procedural",
        scope="agent",
        user_id="u-123",
        content="Always check visa requirements before searching flights.",
        source="reflection",
        importance_score=0.85,
    )

    state = AgentState(user_id="u-123")

    # Multi-turn conversation
    turns = [
        "What units should I use for my workout tracker?",
        "Book me a flight to Tokyo next month.",
        "Suggest some snacks for my road trip.",
    ]

    for i, user_msg in enumerate(turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_msg}")
        print(f"Memory context:\n{state.memory_context or '  (first turn)'}")

        response = await run_agent_turn(state, client, user_msg)
        print(f"Assistant: {response}")

    # Show accumulated memories
    print("\n--- All stored memories ---")
    all_memories = await client.list(filters={"user_id": "u-123"})
    for mem in all_memories:
        print(f"  [{mem.type:10s}] {mem.content[:80]}")

    # Show attribution for the last turn
    print("\n--- Attribution ---")
    memories = await client.search(
        "snacks for road trip",
        filters={"user_id": "u-123"},
        top_k=3,
    )
    attribution = client.build_attribution(memories)
    print(f"  Memories used: {attribution.memories_used}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — Real LangGraph integration (requires langgraph + langchain)
# ═══════════════════════════════════════════════════════════════════════
# Uncomment and run with:
#   pip install engram langgraph langchain-openai

"""
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from engramx import MemoryClient

# --- Setup ---
engram = MemoryClient(driver="sqlite")
model = ChatOpenAI(model="gpt-4o-mini")


# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    memory_context: str
    user_id: str


# --- Nodes ---
async def recall_memories(state: AgentState) -> dict:
    user_msg = state["messages"][-1].content if state["messages"] else ""
    user_id = state.get("user_id", "default")

    memories = await engram.search(
        query=user_msg,
        filters={"user_id": user_id},
        top_k=5,
    )
    context = "\\n".join(f"- {m.record.content}" for m in memories) or "None"
    return {"memory_context": context}


async def call_llm(state: AgentState) -> dict:
    system = SystemMessage(content=(
        "You are a helpful assistant with memory.\\n"
        f"Relevant memories:\\n{state.get('memory_context', 'None')}"
    ))
    response = await model.ainvoke([system] + state["messages"])
    return {"messages": [response]}


async def persist_memories(state: AgentState) -> dict:
    user_id = state.get("user_id", "default")
    msgs = state["messages"]
    if len(msgs) >= 2:
        await engram.add(
            type="episodic",
            scope="user",
            user_id=user_id,
            content=f"User: {msgs[-2].content}\\nAssistant: {msgs[-1].content}",
            source="conversation",
            importance_score=0.7,
        )
    return {}


# --- Build graph ---
graph = StateGraph(AgentState)
graph.add_node("recall", recall_memories)
graph.add_node("llm", call_llm)
graph.add_node("persist", persist_memories)

graph.add_edge(START, "recall")
graph.add_edge("recall", "llm")
graph.add_edge("llm", "persist")
graph.add_edge("persist", END)

# Compile with checkpointer for short-term state
checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)


# --- Run ---
async def run_langgraph_demo():
    config = {"configurable": {"thread_id": "session-001"}}

    # Turn 1
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="What units should I use?")],
            "user_id": "u-123",
        },
        config=config,
    )
    print(f"Assistant: {result['messages'][-1].content}")

    # Turn 2 — same thread, agent remembers via checkpoints + Engram
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="Book a flight to Tokyo.")],
            "user_id": "u-123",
        },
        config=config,
    )
    print(f"Assistant: {result['messages'][-1].content}")

    # New thread — short-term state is fresh, but Engram long-term memory persists
    new_config = {"configurable": {"thread_id": "session-002"}}
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="What do you remember about me?")],
            "user_id": "u-123",
        },
        config=new_config,
    )
    print(f"Assistant: {result['messages'][-1].content}")


# asyncio.run(run_langgraph_demo())
"""


# ═══════════════════════════════════════════════════════════════════════
# Run Part 1
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    asyncio.run(demo_mock_langgraph())
