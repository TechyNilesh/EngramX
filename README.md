<p align="center">
  <img
    src="assets/engram_logo.png"
    alt="Engram logo"
    width="220"
  />
</p>

<p align="center"><strong>The Open-Source Agent Memory Framework</strong></p>

<p align="center">
  <a href="https://github.com/TechyNilesh/engram">
    <img src="https://img.shields.io/github/last-commit/TechyNilesh/engram?style=for-the-badge" alt="Last commit" />
  </a>
  <a href="https://github.com/TechyNilesh/engram/stargazers">
    <img src="https://img.shields.io/github/stars/TechyNilesh/engram?style=for-the-badge" alt="GitHub stars" />
  </a>
  <a href="https://pypi.org/project/engram/">
    <img src="https://img.shields.io/pepy/dt/engram?style=for-the-badge" alt="Total downloads" />
  </a>
  <img src="https://img.shields.io/badge/status-alpha-orange?style=for-the-badge" alt="Project status" />
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white&style=for-the-badge" alt="Python 3.10+" />
</p>

---

## Why Engram?

An **engram** (from the Greek *engramma*, "that which is written on") is the neuroscience term for the physical trace a memory leaves in the brain. Engram brings the same idea to AI agents: structured, durable, queryable memory that works across any framework or backend.

The current ecosystem for agent memory is fragmented, framework-locked, and under-tooled. Every platform defines memory differently, uses incompatible schemas, and provides minimal support for debugging, governance, or cross-framework composability. **Engram is the neutral abstraction layer that sits under or beside any of them.**

## How Engram Compares

| Dimension | Mem0 | Zep | Letta/MemGPT | LangChain | **Engram** |
|---|---|---|---|---|---|
| Common memory schema | Proprietary | Proprietary | Proprietary | Per-pattern | **Canonical** |
| Episodic / Semantic / Procedural | Partial | Partial | Partial (tiers) | Partial | **First-class** |
| Pluggable backends | Limited | No | No | Partial | **9 drivers** |
| Declarative lifecycle policies | Partial | No | No | No | **YAML DSL** |
| Retrieval explainability | Dashboard | No | ADE only | No | **Built-in API** |
| Right-to-be-forgotten | Cloud tier | No | No | No | **Policy-driven** |
| Framework adapters | SDK only | API only | No | Native only | **LangChain, LlamaIndex, AutoGen** |
| Built-in test harness | No | No | No | No | **`MemoryHarness`** |
| Composable with others | No | No | No | No | **Wraps Mem0/Zep** |

---

## Features

- **Canonical schema** — `MemoryRecord` with episodic, semantic, procedural, and meta types
- **9 storage backends** — SQLite, PostgreSQL + pgvector, ChromaDB, Qdrant, Redis, Neo4j, Mem0, Zep, and in-memory
- **Policy engine** — Declarative YAML rules for extraction, retention decay, summarization/promotion, and GDPR governance
- **Observability** — Retrieval traces, memory timelines, output attribution, and `explain()` as a first-class API
- **Hybrid retrieval** — Lexical + vector + importance + decay scoring with configurable modes
- **Embedding providers** — OpenAI, Gemini, Cohere, Mistral, Together AI, LiteLLM (100+ providers), Sentence Transformers, and built-in hash embedder
- **LLM reflection** — Pluggable reflectors (OpenAI, Anthropic, LiteLLM, or any OpenAI-compatible provider via `base_url`) for summarizing episodic memory into durable knowledge
- **Framework adapters** — Thin integrations for LangChain, LlamaIndex, and AutoGen
- **Agent runner** — `EngramAgent` with `run_with_attribution()` for tracking which memories influenced output
- **Background jobs** — `PolicyJobScheduler` for scheduled decay, promotion, and governance enforcement
- **Test harness** — `MemoryHarness` for testing memory behavior in CI
- **Async-first** — Full async API with sync wrappers for convenience

---

## Installation

```bash
pip install engram
```

Install with optional backends:

```bash
pip install engram[postgres]           # PostgreSQL + pgvector
pip install engram[chroma]             # ChromaDB
pip install engram[qdrant]             # Qdrant
pip install engram[redis]              # Redis
pip install engram[neo4j]              # Neo4j
pip install engram[mem0]               # Mem0 delegation
pip install engram[zep]                # Zep delegation
pip install engram[openai]             # OpenAI embeddings + reflection
pip install engram[anthropic]          # Anthropic reflection
pip install engram[litellm]            # LiteLLM (100+ providers)
pip install engram[sentence-transformers]  # Local embeddings
pip install engram[langchain]          # LangChain adapter
pip install engram[llamaindex]         # LlamaIndex adapter
pip install engram[autogen]            # AutoGen adapter
pip install engram[all]                # Everything
```

Install from source:

```bash
git clone https://github.com/TechyNilesh/engram.git
cd engram
pip install -e .[dev]
```

---

## Quick Start

```python
import asyncio
from engram import MemoryClient

async def main() -> None:
    client = MemoryClient(driver="memory")

    # Store a memory
    memory_id = await client.add(
        type="semantic",
        scope="user",
        user_id="u-123",
        content="User prefers metric units for measurements.",
        source="conversation",
        importance_score=0.8,
        sensitivity_level="internal",
    )

    # Search memories
    results = await client.search(
        query="unit preferences",
        filters={"scope": "user", "user_id": "u-123", "type": "semantic"},
        top_k=5,
    )

    # Explain retrieval
    trace = await client.explain(
        query="unit preferences",
        filters={"user_id": "u-123"},
    )

    print(results[0].record.content)
    print(trace.to_markdown())

asyncio.run(main())
```

---

## Storage Backends

Engram decouples the API from storage via a `BaseDriver` interface. All 9 drivers implement the same contract.

| Driver | Backend | Best for | Install extra |
|---|---|---|---|
| `InMemoryDriver` | Python dict | Tests, prototyping | — |
| `SQLiteDriver` | SQLite | Local dev, single-agent | — |
| `PostgresDriver` | PostgreSQL + pgvector | Production, vector search | `postgres` |
| `ChromaDriver` | ChromaDB | Lightweight semantic search | `chroma` |
| `QdrantDriver` | Qdrant | High-performance vector search | `qdrant` |
| `RedisDriver` | Redis / Valkey | Fast short-term memory, caching | `redis` |
| `Neo4jDriver` | Neo4j | Graph-structured memory, temporal edges | `neo4j` |
| `Mem0Driver` | Mem0 API | Delegate to Mem0 + Engram observability | `mem0` |
| `ZepDriver` | Zep API | Delegate to Zep's temporal knowledge graph | `zep` |

```python
from engram import MemoryClient

# Use any driver by name
client = MemoryClient(driver="sqlite")
client = MemoryClient(driver="postgres")
client = MemoryClient(driver="chroma")
client = MemoryClient(driver="qdrant")
client = MemoryClient(driver="redis")
client = MemoryClient(driver="neo4j")

# Or pass a driver instance directly
from engram import PostgresDriver
client = MemoryClient(PostgresDriver(dsn="postgresql://localhost/engram"))
```

---

## Memory Types

Engram enforces meaningful separation of three cognitive memory types:

**Episodic** — Raw experiences: conversation turns, tool calls, agent decisions.

```python
await client.add(type="episodic", scope="session", content="Payment API returned 402.", source="tool_call")
```

**Semantic** — Distilled facts: user preferences, domain knowledge, stable truths.

```python
await client.add(type="semantic", scope="user", content="User prefers metric units.", source="reflection")
```

**Procedural** — Learned skills: action templates and decision heuristics from repeated success.

```python
await client.add(type="procedural", scope="agent", content="Check visa before searching flights.", source="reflection")
```

---

## Core API

All methods are async-first with sync wrappers (`add_sync`, `search_sync`).

| Method | Description |
|---|---|
| `add()` | Store a memory |
| `search()` | Hybrid search (lexical + vector + importance + decay) |
| `get()` | Retrieve by ID (with access control) |
| `update()` | Patch memory fields |
| `delete()` | Delete by filter expressions (GDPR bulk delete) |
| `explain()` | Retrieval trace with scores, decay factors, matched terms |
| `timeline()` | Ordered history of all memory operations |
| `ingest_event()` | Process a signal through the policy engine |
| `promote()` | Promote episodic patterns to procedural memory |
| `apply_decay()` | Apply retention decay to all memories |
| `forget_user()` | Right-to-be-forgotten workflow |
| `run_with_attribution()` | Execute task and track which memories influenced the output |

---

## Policy Configuration

Lifecycle policies are declarative YAML — no imperative code required.

```yaml
# engram.yaml
driver:
  kind: sqlite
  path: .engram/engram.db

policies:
  extraction:
    - name: extract-user-preferences
      trigger: conversation_turn
      conditions:
        - content_matches: [prefer, always, never, like]
      create:
        type: semantic
        scope: user
        importance_score: 0.8
        sensitivity_level: internal

  retention:
    - name: session-episode-decay
      applies_to: { type: episodic, scope: session }
      decay_function: exponential
      half_life_days: 7
      floor_score: 0.1

  summarization:
    - name: promote-repeated-success
      applies_to: { type: episodic }
      trigger: "same_action_success_count >= 3"
      promote_to:
        type: procedural
        scope: agent
        content: "Confirm prerequisites before executing repeated workflows."

  governance:
    - name: gdpr-user-data
      applies_to:
        scope: user
        sensitivity_level: [confidential, restricted]
      retention_days: 30
      on_user_deletion: delete_all
      audit_log: true
```

```python
from engram import MemoryClient
client = MemoryClient(config="engram.yaml")
```

---

## Embedding Providers

Engram ships with a dependency-free hash embedder and supports real embedding models. The OpenAI embedder uses the OpenAI SDK format which works with **any compatible provider** via `base_url`.

```python
from engram import create_embedder

# Default: hash-based (no dependencies, deterministic)
embedder = create_embedder("hash")

# OpenAI
embedder = create_embedder("openai", model="text-embedding-3-small")

# Google Gemini (OpenAI-compatible endpoint)
embedder = create_embedder("openai",
    model="gemini-embedding-001",
    api_key="GEMINI_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Cohere (OpenAI-compatible endpoint)
embedder = create_embedder("openai",
    model="embed-english-v3.0",
    api_key="COHERE_KEY",
    base_url="https://api.cohere.ai/compatibility/v1",
)

# Mistral
embedder = create_embedder("openai",
    model="mistral-embed",
    api_key="MISTRAL_KEY",
    base_url="https://api.mistral.ai/v1",
)

# Together AI
embedder = create_embedder("openai",
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key="TOGETHER_KEY",
    base_url="https://api.together.xyz/v1",
)

# LiteLLM — unified interface for 100+ providers
embedder = create_embedder("litellm", model="cohere/embed-english-v3.0")
embedder = create_embedder("litellm", model="gemini/gemini-embedding-001")
embedder = create_embedder("litellm", model="bedrock/amazon.titan-embed-text-v1")

# Sentence Transformers (local models)
embedder = create_embedder("sentence_transformers", model_name="all-MiniLM-L6-v2")

# Pass to client
client = MemoryClient(driver="memory", embedder=embedder)
```

| Provider | Via | Install |
|---|---|---|
| OpenAI | `create_embedder("openai")` | `pip install engram[openai]` |
| Google Gemini | `create_embedder("openai", base_url=...)` | `pip install engram[openai]` |
| Cohere | `create_embedder("openai", base_url=...)` | `pip install engram[openai]` |
| Mistral | `create_embedder("openai", base_url=...)` | `pip install engram[openai]` |
| Together AI | `create_embedder("openai", base_url=...)` | `pip install engram[openai]` |
| LiteLLM (100+) | `create_embedder("litellm")` | `pip install engram[litellm]` |
| Sentence Transformers | `create_embedder("sentence_transformers")` | `pip install engram[sentence-transformers]` |
| Hash (local, no deps) | `create_embedder("hash")` | — |

---

## LLM Reflection

Summarize episodic memory into durable knowledge using LLM-powered reflection:

```python
from engram import create_reflector

# OpenAI
reflector = create_reflector("openai", model="gpt-4o-mini")

# Any OpenAI-compatible provider via base_url
reflector = create_reflector("openai",
    model="gemini-2.0-flash",
    api_key="GEMINI_KEY",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Anthropic
reflector = create_reflector("anthropic", model="claude-sonnet-4-20250514")

# LiteLLM — any of 100+ providers
reflector = create_reflector("litellm", model="anthropic/claude-sonnet-4-20250514")
reflector = create_reflector("litellm", model="gemini/gemini-2.0-flash")
reflector = create_reflector("litellm", model="cohere/command-r-plus")
```

---

## Framework Adapters

### LangChain

```python
from engram import MemoryClient
from engram.adapters import EngramChatMemory

client = MemoryClient(driver="sqlite")
memory = EngramChatMemory(client=client, user_id="u-123")

# Works as a LangChain memory drop-in
context = memory.load_memory_variables({"input": "What units do I prefer?"})
memory.save_context({"input": "Use metric"}, {"output": "Noted!"})
```

### LlamaIndex

```python
from engram import MemoryClient
from engram.adapters import EngramMemoryBlock

client = MemoryClient(driver="sqlite")
block = EngramMemoryBlock(client=client, user_id="u-123")

context = block.get("unit preferences")
block.put("User prefers metric units.")
```

### AutoGen

```python
from engram import MemoryClient
from engram.adapters import EngramAutoGenMemory

client = MemoryClient(driver="sqlite")
memory = EngramAutoGenMemory(client=client, user_id="u-123")

results = memory.query("payment preferences")
memory.add("User prefers credit card payments.")
```

---

## Agent with Attribution

Track which memories influenced agent output:

```python
from engram import MemoryClient

client = MemoryClient(driver="memory")

async def my_llm(prompt, memories):
    return "Response based on memories"

response, attribution = await client.run_with_attribution(
    "Book a flight to Tokyo",
    user_id="u-123",
    runner=my_llm,
)

print(attribution.memories_used)    # ["mem-abc", "mem-def"]
print(attribution.policies_fired)   # ["extract-user-preferences"]
```

Or use the standalone `EngramAgent` with OpenAI:

```python
from engram import MemoryClient, EngramAgent

client = MemoryClient(driver="sqlite")
agent = EngramAgent(client=client, model="gpt-4o-mini")

response, attribution = await agent.run_with_attribution(
    "What units should I use?",
    user_id="u-123",
)
```

---

## Observability

Every retrieval is traceable:

```python
trace = await client.explain(query="unit preferences", filters={"user_id": "u-123"})
print(trace.to_markdown())
```

Output:

```
# Memory Trace
- Query: `unit preferences`
- Filters: `{'user_id': 'u-123'}`

| Memory ID | Score | Decay Adjusted | Matched Terms | Policy Filters |
| --- | ---: | ---: | --- | --- |
| `mem-abc-123` | 0.870 | 0.740 | metric, unit | scope=user |
```

Timeline of all operations:

```python
events = await client.timeline(user_id="u-123", types=["episodic", "semantic"])
for event in events:
    print(f"{event.timestamp} {event.operation} {event.memory_id}")
```

---

## Background Jobs

Schedule automatic decay, promotion, and governance enforcement:

```python
from engram import MemoryClient, PolicyJobScheduler

client = MemoryClient(config="engram.yaml")
scheduler = PolicyJobScheduler(
    driver=client.driver,
    policy_engine=client.policy_engine,
    interval_seconds=300,  # Run every 5 minutes
)

# Run once
report = await scheduler.run_once()
print(f"Decayed: {len(report.decay_updated_ids)}")
print(f"Promoted: {len(report.promoted_ids)}")
print(f"Deleted: {len(report.deleted_ids)}")

# Or run continuously
await scheduler.start()
# ... later ...
await scheduler.stop()
```

---

## Chat Example

Example chat loop with persistent memory:

```python
from openai import OpenAI
from engram import MemoryClient

openai_client = OpenAI()
memory = MemoryClient(driver="sqlite")

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    relevant = memory.search_sync(query=message, filters={"user_id": user_id}, top_k=3)
    memories_str = "\n".join(f"- {m.record.content}" for m in relevant) or "- None yet."

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful AI.\nUser Memories:\n{memories_str}"},
            {"role": "user", "content": message},
        ],
    )
    reply = response.choices[0].message.content or ""

    memory.add_sync(
        type="episodic", scope="user", user_id=user_id,
        content=f"User: {message}\nAssistant: {reply}",
        source="conversation", importance_score=0.7,
    )
    return reply

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        break
    print(f"AI: {chat_with_memories(user_input)}")
```

---

## Testing

```bash
pip install -e .[dev]
python -m pytest -q
```

The test suite covers storage drivers, the policy engine, test harness, adapters, background jobs, retrieval features, and optional driver dependency checks.

---

## Development

```bash
git clone https://github.com/TechyNilesh/engram.git
cd engram
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m pytest -q
```

---

## Architecture

```
engram/
  schema.py              # MemoryRecord, MemoryType, MemoryScope, SensitivityLevel
  client.py              # MemoryClient — unified API
  config.py              # YAML config loading
  policy.py              # PolicyEngine — extraction, decay, promotion, governance
  lifecycle.py           # Lifecycle functions — extract, decay, summarize, governance
  observability.py       # MemoryTrace, ScoredMemory, OutputAttribution, MemoryTimelineEvent
  embedding.py           # Hash, OpenAI, SentenceTransformer embedders
  reflection.py          # LLM reflectors (OpenAI, Anthropic)
  agent.py               # EngramAgent with run_with_attribution()
  jobs.py                # PolicyJobScheduler for background lifecycle jobs
  models.py              # MemorySignal (event model)
  testing.py             # MemoryHarness for CI
  storage/
    base.py              # BaseDriver abstract interface
    memory.py            # InMemoryDriver
    sqlite.py            # SQLiteDriver
    postgres.py          # PostgresDriver (asyncpg + pgvector)
    chroma.py            # ChromaDriver
    qdrant.py            # QdrantDriver
    redis.py             # RedisDriver
    neo4j.py             # Neo4jDriver
    mem0.py              # Mem0Driver (delegation)
    zep.py               # ZepDriver (delegation)
    ranking.py           # Hybrid scoring (lexical + vector + importance + decay)
  adapters/
    langchain.py         # EngramChatMemory
    llamaindex.py        # EngramMemoryBlock
    autogen.py           # EngramAutoGenMemory
    chat.py              # EngramChatAdapter
```

---

## Core Contributor

<a href="https://github.com/TechyNilesh">
  <img
    src="https://github.com/TechyNilesh.png?size=160"
    alt="Nilesh Verma"
    width="60"
    style="border-radius: 0%;"
  />
</a>

**Nilesh Verma**

---

## Citation

```bibtex
@software{verma2026engram,
  author = {Nilesh Verma},
  title = {Engram: The Open-Source Agent Memory Framework},
  year = {2026},
  url = {https://github.com/TechyNilesh/engram},
  version = {0.1.0}
}
```

---

## License

MIT
