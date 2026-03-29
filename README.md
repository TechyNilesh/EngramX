<p align="center">
  <img
    src="https://raw.githubusercontent.com/TechyNilesh/engram/main/engram_logo.png"
    alt="Engram logo"
    width="220"
  />
</p>

<p align="center">
  <strong>The Open-Source Agent Memory Framework</strong>
</p>

<p align="center">
  <a href="https://github.com/TechyNilesh/engram">
    <img src="https://img.shields.io/badge/GitHub-TechyNilesh%2Fengram-181717?logo=github" alt="GitHub repository" />
  </a>
  <a href="https://github.com/TechyNilesh/engram">
    <img src="https://img.shields.io/badge/status-MVP-2ea44f" alt="Project status" />
  </a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" />
</p>

## Features

- Canonical `MemoryRecord` schema for episodic, semantic, procedural, and meta memory
- Async-first `MemoryClient` with simple sync wrappers
- In-memory and SQLite drivers
- Policy engine for extraction, retention decay, governance deletion, and procedural promotion
- Retrieval traces and memory timelines for observability
- Test harness for memory behavior in CI

## Installation

### Install from PyPI

```bash
pip install engram
```

### Install the latest version from GitHub

```bash
pip install "engram @ git+https://github.com/TechyNilesh/engram.git"
```

### Install from a local clone

```bash
git clone https://github.com/TechyNilesh/engram.git
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick Start

```python
import asyncio

from engram import MemoryClient


async def main() -> None:
    client = MemoryClient(driver="memory")

    memory_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User prefers metric units for measurements.",
        source="conversation",
        importance_score=0.8,
        sensitivity_level="internal",
    )

    results = await client.search(
        query="unit preferences",
        filters={"scope": "user", "user_id": "u-123", "type": "semantic"},
        top_k=5,
    )

    trace = await client.explain(
        query="unit preferences",
        filters={"user_id": "u-123"},
    )

    print(memory_id)
    print(results[0].record.content)
    print(trace.to_markdown())


asyncio.run(main())
```

## Chat Example

This is the same pattern as a simple Mem0-style chat loop, but using Engram:

```python
from openai import OpenAI

from engram import MemoryClient

openai_client = OpenAI()
memory = MemoryClient(driver="sqlite")


def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    relevant_memories = memory.search_sync(
        query=message,
        filters={"user_id": user_id},
        top_k=3,
    )
    memories_str = "\n".join(f"- {entry.record.content}" for entry in relevant_memories) or "- None yet."

    system_prompt = (
        "You are a helpful AI. Answer the question based on the query and memories.\n"
        f"User Memories:\n{memories_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        messages=messages,
    )
    assistant_response = response.choices[0].message.content or ""

    memory.add_sync(
        type="episodic",
        scope="user",
        user_id=user_id,
        content=f"User: {message}\nAssistant: {assistant_response}",
        source="conversation",
        importance_score=0.7,
        sensitivity_level="internal",
    )

    return assistant_response


def main() -> None:
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"AI: {chat_with_memories(user_input)}")


if __name__ == "__main__":
    main()
```

## Policy Configuration

Create an `engram.yaml` file:

```yaml
driver:
  kind: sqlite
  path: .engram/engram.db

policies:
  extraction:
    - name: extract-user-preferences
      trigger: conversation_turn
      conditions:
        - content_matches:
            - prefer
            - always
            - never
            - like
      create:
        type: semantic
        scope: user
        importance_score: 0.8
        sensitivity_level: internal

  retention:
    - name: session-episode-decay
      applies_to:
        type: episodic
        scope: session
      decay_function: exponential
      half_life_days: 7
      floor_score: 0.1

  summarization:
    - name: promote-repeated-success
      applies_to:
        type: episodic
      trigger: "same_action_success_count >= 3"
      promote_to:
        type: procedural
        scope: agent
        content: "Confirm prerequisites before executing repeated workflows."

  governance:
    - name: gdpr-user-data
      applies_to:
        scope: user
        sensitivity_level:
          - confidential
          - restricted
      retention_days: 30
      on_user_deletion: delete_all
      audit_log: true
```

Then load it:

```python
from engram import MemoryClient

client = MemoryClient(config="engram.yaml")
```

## Core API

```python
from engram import MemoryClient

client = MemoryClient(driver="memory")

memory_id = client.add_sync(
    type="episodic",
    scope="session",
    session_id="s-1",
    content="Agent attempted payment and received a 402 error.",
    source="tool_call",
    importance_score=0.85,
    sensitivity_level="internal",
)

results = client.search_sync(
    "payment failure",
    filters={"scope": "session", "type": "episodic"},
)
```

Async methods:

- `add()`
- `search()`
- `list()`
- `get()`
- `update()`
- `delete()`
- `explain()`
- `timeline()`
- `ingest_event()`
- `promote()`
- `apply_decay()`
- `forget_user()`

## Testing

```bash
python -m pytest -q
```

The repository ships with coverage for storage behavior, the test harness, and policy engine workflows.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
```

## Repository

- Source: [https://github.com/TechyNilesh/engram](https://github.com/TechyNilesh/engram)
- Install latest from GitHub: `pip install "engram @ git+https://github.com/TechyNilesh/engram.git"`

## Core Contributor

<p align="center">
  <a href="https://github.com/TechyNilesh">
    <img
      src="https://github.com/TechyNilesh.png?size=160"
      alt="Nilesh Verma"
      width="120"
      style="border-radius: 50%;"
    />
  </a>
</p>

<p align="center">
  <strong>Nilesh Verma</strong><br />
  Core Contributor
</p>

## Citation

If you use Engram in research, tooling, or internal platforms, cite the GitHub repository:

```bibtex
@software{verma2026engram,
  author = {Nilesh Verma},
  title = {Engram: The Open-Source Agent Memory Framework},
  year = {2026},
  url = {https://github.com/TechyNilesh/engram},
  version = {0.1.0}
}
```
