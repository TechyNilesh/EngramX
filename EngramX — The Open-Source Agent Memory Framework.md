# Engram — The Open-Source Agent Memory Framework

> **"Memory is not a byproduct of intelligence. It is its foundation."**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)]()
[![Status: Proposal / RFC](https://img.shields.io/badge/Status-Proposal%20%2F%20RFC-yellow)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-green)]()

---

## Why This Name?

An **engram** (from the Greek *engramma*, "that which is written on") is the scientific term for the physical trace a memory leaves in the brain — the actual biological record of an experience. It is precise, neuroscience-grounded, and unused in the current LLM tooling ecosystem. Just as engrams encode experiences as durable neural patterns, **Engram** encodes agent experiences as structured, queryable, policy-governed memory — making AI agents that truly learn from their past.

**Alternative names considered:**
| Name | Origin | Vibe |
|------|--------|-------|
| **Engram** ⭐ *(recommended)* | Neuroscience — memory trace in the brain | Precise, scientific, memorable |
| Mneme | Greek Muse of memory | Elegant, minimal |
| Loci | "Method of Loci" (memory palace) | Spatial, classical |
| Eidetic | Perfect photographic recall | Aspirational |
| Palimpsest | Layered, rewritten manuscript | Poetic, layered memory metaphor |

---

## 1. Executive Summary

The current open-source ecosystem for LLM agent memory is fragmented, framework-locked, and under-tooled. Every major platform — Mem0, Zep, Letta/MemGPT, LangChain, LlamaIndex, AutoGen — defines memory differently, uses incompatible schemas, and provides minimal support for debugging, governance, and cross-framework composability.

**Engram** fills this gap with a framework-agnostic, standards-first memory layer that:

- Defines a **canonical memory schema** covering episodic, semantic, and procedural types
- Provides **pluggable backend drivers** (SQL, vector DB, knowledge graph, KV store)
- Enforces **declarative lifecycle policies** (extraction, decay, summarization, privacy)
- Ships a **first-class observability toolkit** for inspecting, tracing, and testing memory behavior
- Offers **thin integration adapters** for LangChain, LlamaIndex, AutoGen, and custom agent loops

Engram is not a competitor to Mem0, Zep, or Letta — it is the neutral abstraction layer that can sit **under or beside** any of them.

---

## 2. The Problem: A Fragmented Memory Ecosystem

Despite rapid growth in LLM agents, memory remains an afterthought glued on per-project. The core issues are:

### 2.1 No Common Schema or Model

Each system invents its own concept of "memory":

| System | Memory Concept | Lock-in |
|--------|---------------|---------|
| Mem0 | Multi-level KV + vector hybrid | Mem0 API / SDK |
| Zep | Temporal knowledge graph (Graphiti) | Zep server + Graphiti library |
| Letta/MemGPT | OS-style tiered pages (core/recall/archival) | Letta runtime |
| LangChain | Buffer / summary / entity / vector / KG patterns | LangChain abstractions |
| LlamaIndex | Composable memory blocks (static, fact, vector) | LlamaIndex agents |
| AutoGen | Pluggable backends (Chroma, Redis, Mem0, Zep) | AutoGen agents |

There is no shared vocabulary. You cannot move a "memory" from Letta to a LangChain agent without rewriting integration code.

### 2.2 Missing First-Class Support for Memory Types

Blogs and research consistently describe agent memory as episodic + semantic + procedural, but tools do not enforce or leverage these distinctions:

- Episodic event logs, semantic facts, and procedural skills are often mixed in one vector store
- No declarative policy language to say: _"summarize successful patterns into semantic rules after N repetitions"_
- Procedural memory is left entirely to external code or tool registries — it is not a first-class memory citizen

### 2.3 Black-Box Memory Behavior

Production teams cannot answer basic questions like:
- _Why_ did the agent retrieve this memory and not that one?
- _Which memories_ influenced this particular output?
- _What_ was stored three sessions ago, and has it decayed?

Letta provides an Agent Development Environment (ADE), and Mem0 offers cloud dashboards — but no **backend-agnostic, open-source** observability standard exists.

### 2.4 No Standard Governance or Privacy Layer

Persistent memory introduces real compliance risk. Today's tools expose low-level CRUD, leaving policy entirely to the application:

- No standard way to tag memories with sensitivity labels or retention schedules
- No built-in "right to be forgotten" workflow that works across backends
- No access audit trail in a portable format

### 2.5 Retrieval-Only Benchmarks Miss the Full Picture

LOCOMO, Deep Memory Retrieval (DMR), and LongMemEval measure retrieval accuracy and latency — valuable, but narrow. They do not measure how memory design affects end-to-end task success, user trust, or the value of different lifecycle policies over long horizons.

### 2.6 Memory as Static Storage, Not a Learning Asset

Current tools store and retrieve. None provide principled pipelines to:
- Mine episodic logs for emerging semantic rules
- Detect clusters of failure patterns and route them to evaluation pipelines
- Schedule background "reflection" jobs that convert repeated experiences into durable knowledge

---

## 3. Solution: Engram Architecture

### 3.1 Design Principles

1. **Schema-first** — define the canonical memory record before touching any storage engine
2. **Policy-driven** — lifecycle rules (extraction, retention, decay, privacy) are declarative configuration, not imperative code
3. **Backend-agnostic** — the same API surface works whether the store is SQLite, pgvector, Neo4j, or Redis
4. **Observability-native** — every memory operation is traceable from day one
5. **Framework-neutral** — thin adapters make Engram feel native in LangChain, LlamaIndex, AutoGen, and raw Python loops
6. **Composable with existing tools** — Engram can delegate storage to Mem0 or Zep while adding its schema, policies, and observability on top

---

### 3.2 Core Memory Schema

Every memory in Engram is a structured record conforming to this canonical schema:

```python
from dataclasses import dataclass, field
from typing import Literal, Optional, Any
from datetime import datetime

MemoryType  = Literal["episodic", "semantic", "procedural", "meta"]
MemoryScope = Literal["user", "session", "agent", "global"]
SensitivityLevel = Literal["public", "internal", "confidential", "restricted"]

@dataclass
class MemoryRecord:
    # Identity
    id: str                          # UUID v7 (time-ordered)
    type: MemoryType                 # episodic | semantic | procedural | meta

    # Scope and ownership
    scope: MemoryScope               # user | session | agent | global
    agent_id: Optional[str]
    user_id:  Optional[str]
    session_id: Optional[str]

    # Content
    content: str                     # Natural-language text
    payload: Optional[dict]          # Optional structured data (facts, code refs, etc.)

    # Temporal
    timestamp_created: datetime
    timestamp_updated: datetime
    valid_from:  Optional[datetime]  # For temporal/versioned facts (Zep-style)
    valid_to:    Optional[datetime]

    # Provenance
    source: str                      # "conversation" | "tool_call" | "reflection" | "environment"
    provenance: list[str]            # References to source event IDs or log entries
    derived_from: list[str]          # Parent memory IDs (for summarization chains)

    # Retrieval signals
    importance_score: float          # 0.0–1.0, set at creation or by reflection
    access_count: int
    last_accessed: Optional[datetime]
    decay_factor: float              # 0.0 = fully decayed, 1.0 = fully fresh

    # Governance
    sensitivity_level: SensitivityLevel
    retention_policy: Optional[str] # Policy ID reference
    access_roles: list[str]          # Who/what can read this memory

    # Embedding (populated by storage driver)
    embedding: Optional[list[float]] = field(default=None, repr=False)
    embedding_model: Optional[str]   = None
```

**Why this schema matters:** It is the _contract_ between agents, storage backends, and integration adapters. Any backend driver that ingests and returns `MemoryRecord` objects is automatically compatible with the full Engram ecosystem.

---

### 3.3 Core API Surface

Engram exposes a minimal, consistent API regardless of which backend is active:

```python
from engramx import MemoryClient

client = MemoryClient(config="engram.yaml")  # or programmatic config

# ── Write ──────────────────────────────────────────────────────────────────
await client.add(
    content="User mentioned they prefer metric units for all measurements.",
    type="semantic",
    scope="user",
    user_id="u-123",
    source="conversation",
    importance_score=0.8,
    sensitivity_level="internal",
)

# ── Read (semantic search) ─────────────────────────────────────────────────
results = await client.search(
    query="user unit preferences",
    filters={"scope": "user", "user_id": "u-123", "type": "semantic"},
    top_k=5,
)

# ── Read (temporal / graph query) ─────────────────────────────────────────
history = await client.search(
    query="payment method used last quarter",
    filters={"valid_before": "2025-12-31", "scope": "user"},
    top_k=3,
)

# ── Update ────────────────────────────────────────────────────────────────
await client.update(memory_id="mem-abc", patch={"importance_score": 0.9})

# ── Delete (right to be forgotten) ────────────────────────────────────────
await client.delete(filters={"user_id": "u-123", "sensitivity_level": "confidential"})

# ── Explain retrieval (observability) ─────────────────────────────────────
trace = await client.explain(
    query="user unit preferences",
    filters={"user_id": "u-123"},
)
# Returns: which memories were retrieved, scores, decay factors, policy filters applied
print(trace.to_markdown())
```

**Key design decisions:**
- Async-first (`asyncio`) with sync wrappers for compatibility
- All methods accept `MemoryRecord` objects or plain kwargs for ergonomics
- `explain()` is a first-class method, not an afterthought or debug flag
- `delete()` accepts filter expressions, not just IDs — enabling bulk GDPR compliance flows

---

### 3.4 Pluggable Backend Drivers

Engram decouples the API from storage via a `BaseDriver` interface:

```python
from abc import ABC, abstractmethod

class BaseDriver(ABC):
    @abstractmethod
    async def upsert(self, record: MemoryRecord) -> str: ...

    @abstractmethod
    async def search(
        self,
        query: str,
        filters: dict,
        top_k: int,
    ) -> list[ScoredMemory]: ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord: ...

    @abstractmethod
    async def update(self, memory_id: str, patch: dict) -> MemoryRecord: ...

    @abstractmethod
    async def delete(self, filters: dict) -> int: ...  # Returns count deleted
```

**Planned drivers (in priority order):**

| Driver | Backend | Best for | Status |
|--------|---------|---------|--------|
| `PostgresDriver` | PostgreSQL + pgvector | General-purpose; episodic + semantic; production default | v0.1 |
| `SQLiteDriver` | SQLite + sqlite-vec | Local dev, single-agent scenarios | v0.1 |
| `ChromaDriver` | ChromaDB | Lightweight semantic search, easy setup | v0.2 |
| `QdrantDriver` | Qdrant | High-performance vector search at scale | v0.2 |
| `RedisDriver` | Redis / Valkey | Fast short-term and cache-style memories | v0.2 |
| `Neo4jDriver` | Neo4j | Graph-structured semantic memory + temporal edges | v0.3 |
| `Mem0Driver` | Mem0 (via API) | Delegate to Mem0 while adding Engram schema and observability | v0.3 |
| `ZepDriver` | Zep (via API) | Delegate to Zep's temporal knowledge graph | v0.3 |

**Hybrid retrieval** — The `PostgresDriver` and `Neo4jDriver` support multi-stage retrieval: a vector similarity pass followed by a graph traversal or SQL filter pass, merged via a configurable re-ranking function. This mirrors the power of Zep's Graphiti without requiring a separate service.

---

### 3.5 Memory Types in Practice

Engram enforces meaningful separation of the three cognitive memory types:

#### Episodic Memory
Raw, temporally indexed experiences — conversation turns, tool calls, agent decisions, environment observations.

```yaml
# Example episodic record (auto-created by extraction policy)
type: episodic
scope: session
content: "Agent attempted to book a flight but the payment API returned a 402 error. User was frustrated."
source: tool_call
importance_score: 0.85
provenance: ["event-id-007", "tool-call-id-042"]
valid_from: "2026-03-29T10:15:00Z"
```

**Use:** Retrieved when a similar task is attempted — the agent checks for prior failures before retrying the same approach.

#### Semantic Memory
Decontextualized facts distilled from multiple episodes — user preferences, domain knowledge, stable truths.

```yaml
type: semantic
scope: user
content: "User prefers to receive responses in metric units. Allergic to peanuts. Vegan diet."
source: reflection
derived_from: ["ep-001", "ep-002", "ep-007"]
importance_score: 0.95
valid_from: "2026-01-01T00:00:00Z"
valid_to: null  # Persists until invalidated
```

**Use:** Injected as standing context at the start of every session, without retrieving raw conversation history.

#### Procedural Memory
Learned skills, reusable action templates, and decision heuristics derived from repeated successful patterns.

```yaml
type: procedural
scope: agent
content: "When booking travel for the user, always check visa requirements before searching for flights. Confirm dates twice."
source: reflection
derived_from: ["ep-012", "ep-018", "ep-024"]
importance_score: 0.90
payload:
  trigger_pattern: "travel booking"
  action_template: "check_visa_requirements(destination) → search_flights(...)"
```

**Use:** Retrieved when a matching task type is detected, providing the agent with a structured action plan derived from past experience.

---

### 3.6 Policy Engine

Lifecycle policies are declared in YAML and interpreted by Engram's policy engine — no imperative code required for common behaviors.

```yaml
# engram.yaml

policies:
  extraction:
    - name: "log-all-tool-failures"
      trigger: "tool_call_error"
      create:
        type: episodic
        scope: session
        importance_score_formula: "1.0 - (0.1 * retry_count)"
        sensitivity_level: internal

    - name: "extract-user-preferences"
      trigger: "conversation_turn"
      conditions:
        - content_matches: ["prefer", "like", "always", "never", "hate", "love"]
      create:
        type: semantic
        scope: user
        importance_score: 0.8

  retention:
    - name: "session-episode-decay"
      applies_to: {type: episodic, scope: session}
      decay_function: exponential
      half_life_days: 7
      floor_score: 0.1      # Keep forever at low importance once decayed to 0.1

    - name: "user-semantic-persist"
      applies_to: {type: semantic, scope: user}
      decay_function: none   # Semantic facts persist until invalidated

  summarization:
    - name: "promote-repeated-success"
      applies_to: {type: episodic}
      trigger: "same_action_success_count >= 3"
      promote_to:
        type: procedural
        scope: agent
        summary_model: "gpt-4o-mini"   # Configurable LLM for reflection

  governance:
    - name: "gdpr-user-data"
      applies_to: {scope: user, sensitivity_level: [confidential, restricted]}
      retention_days: 30
      on_user_deletion: delete_all    # Supports "right to be forgotten"
      audit_log: true
```

**Policy engine capabilities:**

- **Extraction triggers** — fire on conversation turns, tool calls, environment events, or custom signals
- **Decay functions** — exponential, linear, step, or none; configurable half-life per memory type
- **Summarization and promotion** — background jobs that convert episodic clusters into semantic rules or procedural skills
- **Governance rules** — retention schedules, sensitivity tagging, access control, and deletion workflows
- **Pluggable LLMs** — reflection and summarization steps can use any configured LLM client

---

### 3.7 Observability and Debugging Toolkit

Engram ships a structured observability layer as a core module, not an optional add-on.

#### Retrieval Traces

Every `search()` call can return a trace object:

```python
trace = await client.explain(query="user payment preference", filters={...})

print(trace.query)             # "user payment preference"
print(trace.retrieved[0].id)   # "mem-abc-123"
print(trace.retrieved[0].score)            # 0.87 (cosine similarity)
print(trace.retrieved[0].decay_adjusted)   # 0.74 (after decay factor)
print(trace.retrieved[0].policy_filters)   # ["session-scope", "user-u-123"]
print(trace.to_markdown())     # Human-readable retrieval explanation
```

#### Memory Timeline

A per-user/agent/session timeline of all memory operations:

```python
timeline = await client.timeline(
    user_id="u-123",
    from_dt="2026-03-01",
    to_dt="2026-03-29",
    types=["episodic", "semantic"],
)
# Returns ordered list of MemoryEvent objects with operation, memory_id, timestamp, actor
```

#### Attribution Links

Attach memory IDs to agent outputs, enabling backward tracing:

```python
# When generating a response:
response, attribution = await agent.run_with_attribution(task="Book a flight to Tokyo")

print(attribution.memories_used)   # ["mem-abc", "mem-def"] — which memories influenced this run
print(attribution.policies_fired)  # ["log-all-tool-failures", "extract-user-preferences"]
```

#### Test Harness

Built-in pytest-compatible utilities for testing memory behavior in CI:

```python
from engramx.testing import MemoryHarness

async def test_preference_extraction():
    harness = MemoryHarness(driver="sqlite")  # In-memory for tests
    await harness.inject_conversation([
        ("user", "I always prefer metric units for measurements."),
        ("assistant", "Noted! I'll use metric units going forward."),
    ])
    await harness.run_policies()

    results = await harness.search("unit preferences", type="semantic")
    assert len(results) > 0
    assert "metric" in results[0].content.lower()

async def test_right_to_be_forgotten():
    harness = MemoryHarness(driver="sqlite")
    await harness.seed_memories([...])
    await harness.delete(user_id="u-123")
    remaining = await harness.search("*", filters={"user_id": "u-123"})
    assert len(remaining) == 0
```

---

### 3.8 Framework Integration Adapters

Engram provides thin shims for every major agent framework.

#### LangChain Adapter

```python
from engramx.adapters.langchain import EngramChatMessageHistory, EngramEntityMemory

# Drop-in replacement for LangChain's chat message history
history = EngramChatMessageHistory(
    client=client,
    session_id="sess-001",
    user_id="u-123",
)

# Drop-in replacement for LangChain's entity memory
entity_mem = EngramEntityMemory(
    client=client,
    user_id="u-123",
)

# Use in any LangChain chain or agent as normal
chain = ConversationChain(llm=llm, memory=entity_mem)
```

#### LlamaIndex Adapter

```python
from engramx.adapters.llamaindex import EngramMemoryBlock

# Implements LlamaIndex's MemoryBlock interface
block = EngramMemoryBlock(
    client=client,
    types=["episodic", "semantic"],
    scope="user",
    user_id="u-123",
    top_k=5,
)

agent = ReActAgent.from_tools(tools, memory=Memory.from_blocks([block]))
```

#### AutoGen Adapter

```python
from engramx.adapters.autogen import EngramMemory

# Implements AutoGen's Memory interface
mem = EngramMemory(
    client=client,
    user_id="u-123",
    types=["semantic", "procedural"],
)

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    memory=[mem],
)
```

#### Custom Loop Adapter

For raw agent loops (no framework):

```python
from engramx import MemoryClient, MemoryMiddleware

client = MemoryClient(config="engram.yaml")
middleware = MemoryMiddleware(client=client)

async def agent_step(messages: list, user_id: str) -> str:
    # Retrieve relevant memories and inject into context
    context = await middleware.build_context(messages[-1], user_id=user_id)
    augmented = context.inject(messages)

    # Run LLM
    response = await llm.complete(augmented)

    # Store new memories according to policy
    await middleware.process_output(response, messages[-1], user_id=user_id)

    return response.text
```

---

### 3.9 Benchmarking Harness

Engram ships an evaluation suite to make memory design choices measurable.

```
engram/
└── benchmarks/
    ├── retrieval/
    │   ├── locomo.py        # LOCOMO benchmark adapter
    │   ├── dmr.py           # Deep Memory Retrieval benchmark
    │   └── longmemeval.py   # LongMemEval benchmark
    ├── task/
    │   ├── multi_day_workflow.py   # Task success over long horizons
    │   ├── personalization.py     # User preference recall accuracy
    │   └── failure_learning.py    # Does the agent avoid repeating past errors?
    └── policy/
        ├── decay_impact.py        # How does decay policy affect accuracy?
        └── summarization_value.py # Does reflection improve downstream success?
```

```bash
# Run standard retrieval benchmarks
engram benchmark run --suite locomo --driver postgres --config engram.yaml

# Compare two configurations
engram benchmark compare \
  --config-a engram-vector.yaml \
  --config-b engram-graph.yaml \
  --suite multi_day_workflow

# Output: side-by-side accuracy, latency, token cost, task success rate
```

---

## 4. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.10+ | Ecosystem compatibility; type hints; async |
| Core schema | `dataclasses` + `pydantic` | Validation + serialization |
| Async runtime | `asyncio` | Non-blocking I/O for storage calls |
| Default vector backend | `pgvector` (PostgreSQL) | SQL + vectors in one system; production-grade |
| Local dev backend | `sqlite-vec` | Zero-infrastructure for development |
| Embedding | Configurable (`openai`, `sentence-transformers`, `ollama`) | No lock-in |
| Policy engine | Python + YAML DSL | Human-readable; version-controlled |
| Observability | Structured JSON logs + OpenTelemetry spans | Standard tooling |
| HTTP server (optional) | `FastAPI` + `uvicorn` | For service deployments |
| CLI | `typer` | Intuitive developer experience |
| Tests | `pytest` + `pytest-asyncio` | Standard Python testing |

---

## 5. What Engram is NOT

To keep scope honest:

- **Not a full agent runtime.** Engram does not manage agent loops, tool execution, or prompt construction. Use LangGraph, Letta, AutoGen, or a custom loop for that.
- **Not a managed cloud service (initially).** Engram is a library + optional self-hosted service. Cloud hosting is a community decision after the core is proven.
- **Not a vector database.** Engram orchestrates vector DBs; it does not implement one.
- **Not a replacement for Mem0 or Zep.** Engram can wrap them as backends, extending their power with a unified schema, lifecycle policies, and observability.

---

## 6. How Engram Compares to Existing Tools

| Dimension | Mem0 | Zep | Letta/MemGPT | LangChain Memory | **Engram** |
|-----------|------|-----|-------------|-----------------|-----------|
| Common memory schema | ❌ Proprietary | ❌ Proprietary | ❌ Proprietary | ❌ Per-pattern | ✅ Canonical |
| Episodic / Semantic / Procedural types | Partial | Partial | Partial (tiers) | Partial | ✅ First-class |
| Pluggable backends | Limited | No | No | Partial | ✅ Driver interface |
| Declarative lifecycle policies | Partial | No | No | No | ✅ YAML DSL |
| Retrieval traces / explainability | Dashboard only | No | ADE (runtime-only) | No | ✅ Built-in API |
| Right-to-be-forgotten workflow | Cloud tier | No | No | No | ✅ Policy-driven |
| Framework-agnostic | Mostly | Yes (API) | No | No | ✅ By design |
| Built-in test harness | No | No | No | No | ✅ `MemoryHarness` |
| Benchmark suite | LOCOMO (partial) | DMR, LongMemEval | DMR | No | ✅ Multi-suite |
| Composable with others | Partial | No | No | No | ✅ Wraps Mem0/Zep |

---

## 7. Project Roadmap

### v0.1 — Foundation (Months 1–2)
- [ ] Canonical `MemoryRecord` schema and `MemoryClient` API
- [ ] `PostgresDriver` with pgvector (semantic search + structured filters)
- [ ] `SQLiteDriver` with sqlite-vec (local development)
- [ ] Basic policy engine: extraction + retention + decay
- [ ] Retrieval traces (`explain()`)
- [ ] `MemoryHarness` test utilities
- [ ] LangChain adapter (`EngramChatMessageHistory`, `EngramEntityMemory`)
- [ ] AutoGen adapter (`EngramMemory`)
- [ ] CLI: `engram init`, `engram inspect`, `engram timeline`
- [ ] Docs site with quickstart and API reference

### v0.2 — Ecosystem Expansion (Months 3–4)
- [ ] LlamaIndex adapter (`EngramMemoryBlock`)
- [ ] `ChromaDriver` and `QdrantDriver`
- [ ] `RedisDriver` (fast short-term cache tier)
- [ ] Summarization and reflection policies (background jobs)
- [ ] Memory timeline UI (lightweight web view served by CLI)
- [ ] Attribution links for agent outputs
- [ ] LOCOMO and DMR benchmark adapters
- [ ] Observability: OpenTelemetry span export

### v0.3 — Advanced Memory (Months 5–6)
- [ ] `Neo4jDriver` with temporal edge support (Zep-style graph memory)
- [ ] `Mem0Driver` and `ZepDriver` (delegate + extend existing services)
- [ ] Procedural memory promotion pipeline (episode → skill)
- [ ] Full governance module: audit log, sensitivity tagging, retention enforcement
- [ ] Task-level benchmark scenarios (`multi_day_workflow`, `failure_learning`)
- [ ] Semantic Kernel adapter
- [ ] Plugin architecture for custom memory types and policy triggers

### v1.0 — Production Ready (Month 8+)
- [ ] Stable API with semantic versioning guarantee
- [ ] Security audit
- [ ] Production deployment guide (Docker, Kubernetes)
- [ ] Community governance model
- [ ] Integration with Letta as an archival/recall backend

---

## 8. Repository Structure

```
engram/
├── engram/
│   ├── __init__.py
│   ├── client.py            # MemoryClient — main API surface
│   ├── schema.py            # MemoryRecord, ScoredMemory, MemoryTrace
│   ├── policy/
│   │   ├── engine.py        # Policy interpreter
│   │   ├── extraction.py    # Extraction policy handlers
│   │   ├── retention.py     # Decay + deletion policies
│   │   ├── summarization.py # Reflection + promotion jobs
│   │   └── governance.py    # Privacy, audit, access control
│   ├── drivers/
│   │   ├── base.py          # BaseDriver ABC
│   │   ├── postgres.py
│   │   ├── sqlite.py
│   │   ├── chroma.py
│   │   ├── qdrant.py
│   │   ├── redis.py
│   │   └── neo4j.py
│   ├── adapters/
│   │   ├── langchain.py
│   │   ├── llamaindex.py
│   │   ├── autogen.py
│   │   └── middleware.py    # Custom loop middleware
│   ├── observability/
│   │   ├── traces.py
│   │   ├── timeline.py
│   │   └── otel.py          # OpenTelemetry integration
│   └── testing/
│       └── harness.py       # MemoryHarness
├── benchmarks/
│   ├── retrieval/
│   └── task/
├── docs/
├── examples/
│   ├── langchain_agent.py
│   ├── autogen_team.py
│   └── custom_loop.py
├── tests/
├── engram.yaml.example
├── pyproject.toml
└── README.md
```

---

## 9. Getting Started (Planned API)

```bash
pip install engram
```

```bash
# Initialize a project config
engram init --driver postgres --db-url postgresql://localhost/engram_db

# Inspect memories for a user
engram inspect --user-id u-123

# View memory timeline
engram timeline --user-id u-123 --from 2026-03-01

# Run benchmark
engram benchmark run --suite locomo
```

```python
from engramx import MemoryClient

client = MemoryClient.from_config("engram.yaml")

# Add a memory
await client.add(
    content="User prefers short, bullet-pointed responses.",
    type="semantic",
    scope="user",
    user_id="u-123",
)

# Search
results = await client.search("response format preference", filters={"user_id": "u-123"})
for r in results:
    print(r.content, r.score)

# Explain
trace = await client.explain("response format preference", filters={"user_id": "u-123"})
print(trace.to_markdown())
```

---

## 10. Contributing

Engram is designed as a community-first project. Areas where contributions are most needed:

- **Driver implementations** — new backend drivers (MySQL, Weaviate, Pinecone, etc.)
- **Framework adapters** — bindings for Semantic Kernel, CrewAI, AgentOps, etc.
- **Benchmark scenarios** — real-world task-level evaluation datasets
- **Policy templates** — reusable YAML policy packs for common use cases (customer support, coding assistants, research agents)
- **Documentation and examples** — tutorials, integration guides, notebooks

**Contributing guide:** `CONTRIBUTING.md` (coming with v0.1)
**Discussions and RFC process:** GitHub Discussions
**License:** Apache 2.0

---

## 11. Name & Branding

**Primary name:** `engram`
**PyPI package:** `engram`
**Import:** `import engramx`
**CLI:** `engram`
**GitHub org (suggested):** `engram-ai` or `engram-memory`
**Tagline:** *"Memory for agents that learn."*
**Logo concept:** A minimal ink mark — a single node with radiating edges, evoking a neural engram trace and a knowledge graph simultaneously. Works monochrome. Clean at 16px and 256px.

---

*This document is a living RFC. Open an issue or start a Discussion to propose changes.*

*Engram — built on the gaps left by the tools that came before it.*
