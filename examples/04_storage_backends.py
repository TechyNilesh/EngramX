"""
04 -- Storage Backends
======================

Engram ships with multiple storage drivers so you can pick the backend
that best fits your deployment.  Two drivers work out of the box with
zero external services:

    * InMemoryDriver  -- ephemeral, great for tests and prototyping
    * SQLiteDriver    -- persistent, single-file, no server needed

The remaining drivers (Postgres, ChromaDB, Qdrant, Redis, Neo4j, Mem0,
Zep) require their respective client libraries and a running service.
This example shows full working code for in-memory and SQLite, and
commented constructor patterns for every other backend.

Run:
    python examples/04_storage_backends.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from engramx.client import MemoryClient
from engramx.schema import MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


async def run_crud(client: MemoryClient, label: str) -> None:
    """Exercise basic CRUD operations against whatever driver the client uses."""

    separator(f"{label}: CRUD operations")

    # -- Create --
    id1 = await client.add(
        type="episodic",
        scope="user",
        content="The user prefers dark-mode in every app.",
        user_id="alice",
        importance_score=0.9,
    )
    id2 = await client.add(
        type="semantic",
        scope="global",
        content="Python 3.12 introduced type parameter syntax.",
        importance_score=0.7,
    )
    id3 = await client.add(
        type="episodic",
        scope="user",
        content="The user enjoys hiking on weekends.",
        user_id="alice",
        importance_score=0.6,
    )
    print(f"Created three memories: {id1[:8]}... {id2[:8]}... {id3[:8]}...")

    # -- Read (get by id) --
    record = await client.get(id1)
    print(f"Fetched memory {record.id[:8]}...: {record.content!r}")

    # -- List all --
    all_records = await client.list()
    print(f"Total memories stored: {len(all_records)}")

    # -- Search --
    results = await client.search("dark mode preference", top_k=2)
    print(f"Search for 'dark mode preference' returned {len(results)} result(s):")
    for hit in results:
        print(f"  - [{hit.score:.3f}] {hit.record.content}")

    # -- Update --
    updated = await client.update(id1, {"importance_score": 1.0})
    print(f"Updated importance_score -> {updated.importance_score}")

    # -- Delete --
    deleted_count = await client.delete({"id": id3})
    print(f"Deleted {deleted_count} memory(ies)")

    remaining = await client.list()
    print(f"Memories remaining: {len(remaining)}")


# ---------------------------------------------------------------------------
# 1. InMemory backend -- works out of the box
# ---------------------------------------------------------------------------

async def demo_inmemory() -> None:
    """Ephemeral storage that lives only for the process lifetime."""

    # Shortest way: pass the string "memory" as the driver.
    client = MemoryClient(driver="memory")

    await run_crud(client, "InMemory")


# ---------------------------------------------------------------------------
# 2. SQLite backend -- works out of the box
# ---------------------------------------------------------------------------

async def demo_sqlite() -> None:
    """Persistent storage backed by a single SQLite file."""

    # Use a temporary file so this example is self-contained.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "engram_demo.db"

        # Option A: pass the driver name and configure the path via config.
        client = MemoryClient(config={
            "driver": {"kind": "sqlite", "path": str(db_path)},
        })

        await run_crud(client, "SQLite")

        print(f"\nSQLite file size: {db_path.stat().st_size:,} bytes")
        print(f"Database path:    {db_path}")


# ---------------------------------------------------------------------------
# 3. PostgreSQL  (requires `asyncpg` + a running Postgres instance)
# ---------------------------------------------------------------------------

# To use Postgres, install the extra:
#     pip install engram[postgres]
#
# Then configure the client with a DSN:
#
#     client = MemoryClient(config={
#         "driver": {
#             "kind": "postgres",
#             "path": "postgresql://user:password@localhost:5432/engram",
#         },
#     })
#
# Or instantiate the driver directly:
#
#     from engramx.storage.postgres import PostgresDriver
#     driver = PostgresDriver(
#         dsn="postgresql://user:password@localhost:5432/engram",
#         table_name="engram_memories",   # default
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 4. ChromaDB  (requires `chromadb`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[chroma]
#
# Ephemeral (in-process) ChromaDB:
#
#     client = MemoryClient(config={
#         "driver": {"kind": "chroma"},
#     })
#
# Persistent ChromaDB (on-disk path):
#
#     client = MemoryClient(config={
#         "driver": {"kind": "chroma", "path": "/tmp/chroma_data"},
#     })
#
# Or via the driver class:
#
#     from engramx.storage.chroma import ChromaDriver
#     driver = ChromaDriver(
#         path="/tmp/chroma_data",
#         collection_name="engram_memories",   # default
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 5. Qdrant  (requires `qdrant-client`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[qdrant]
#
# Local in-memory Qdrant (no server needed, but not persistent):
#
#     client = MemoryClient(config={
#         "driver": {"kind": "qdrant"},
#     })
#
# Remote Qdrant instance:
#
#     client = MemoryClient(config={
#         "driver": {"kind": "qdrant", "path": "http://localhost:6333"},
#     })
#
# Or via the driver class:
#
#     from engramx.storage.qdrant import QdrantDriver
#     driver = QdrantDriver(
#         url="http://localhost:6333",
#         collection_name="engram_memories",   # default
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 6. Redis  (requires `redis`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[redis]
#
# Configuration:
#
#     client = MemoryClient(config={
#         "driver": {
#             "kind": "redis",
#             "path": "redis://localhost:6379",
#         },
#     })
#
# Or via the driver class:
#
#     from engramx.storage.redis import RedisDriver
#     driver = RedisDriver(
#         url="redis://localhost:6379",
#         prefix="engram:",   # default key prefix
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 7. Neo4j  (requires `neo4j`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[neo4j]
#
# Configuration:
#
#     client = MemoryClient(config={
#         "driver": {
#             "kind": "neo4j",
#             "path": "bolt://localhost:7687",
#         },
#     })
#
# Or via the driver class:
#
#     from engramx.storage.neo4j import Neo4jDriver
#     driver = Neo4jDriver(
#         uri="bolt://localhost:7687",
#         auth=("neo4j", "neo4j"),   # default
#         database="neo4j",          # default
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 8. Mem0  (requires `mem0ai`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[mem0]
#
# Mem0 uses an API key (set via env var or constructor argument):
#
#     import os
#     os.environ["MEM0_API_KEY"] = "m0-..."
#
#     client = MemoryClient(config={
#         "driver": {"kind": "mem0"},
#     })
#
# Or via the driver class:
#
#     from engramx.storage.mem0 import Mem0Driver
#     driver = Mem0Driver(
#         api_key="m0-...",
#         org_id="org-...",          # optional
#         project_id="proj-...",     # optional
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 9. Zep  (requires `zep-python`)
# ---------------------------------------------------------------------------

# Install:
#     pip install engram[zep]
#
# Zep requires an API key:
#
#     client = MemoryClient(config={
#         "driver": {
#             "kind": "zep",
#             "path": "z_...",       # API key goes in the path field
#         },
#     })
#
# Or via the driver class:
#
#     from engramx.storage.zep import ZepDriver
#     driver = ZepDriver(
#         api_key="z_...",
#         base_url="https://api.getzep.com",   # optional override
#     )
#     client = MemoryClient(driver=driver)


# ---------------------------------------------------------------------------
# 10. YAML config -- switch backends without changing code
# ---------------------------------------------------------------------------

async def demo_yaml_config() -> None:
    """
    You can store your driver choice in a YAML file and load it at startup.
    This makes it easy to use InMemory in tests and Postgres in production
    without touching application code.
    """

    separator("YAML config-driven backend selection")

    # Write a small YAML config to a temp file.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(
            "driver:\n"
            "  kind: memory\n"
            "  path: ':memory:'\n"
        )
        config_path = f.name

    print(f"Config file: {config_path}")
    print(f"Contents:")
    print(f"  driver:")
    print(f"    kind: memory")
    print(f"    path: ':memory:'")

    # Load the client from the YAML config path.
    client = MemoryClient(config=config_path)

    mem_id = await client.add(
        type="semantic",
        scope="global",
        content="Engram supports config-driven backend selection.",
    )
    records = await client.list()
    print(f"\nStored {len(records)} memory via YAML-configured client.")
    print(f"Memory id: {mem_id[:8]}...")

    # To switch to Postgres in production, just change the YAML:
    #
    #   driver:
    #     kind: postgres
    #     path: postgresql://user:password@db-host:5432/engram
    #
    # No application code changes required.

    # Clean up the temp config file.
    Path(config_path).unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    await demo_inmemory()
    await demo_sqlite()
    await demo_yaml_config()

    separator("Summary")
    print("Backends demonstrated live: InMemory, SQLite, YAML config")
    print("Backends shown as patterns: Postgres, ChromaDB, Qdrant,")
    print("                            Redis, Neo4j, Mem0, Zep")
    print()
    print("Install optional extras to unlock additional backends:")
    print("  pip install engram[postgres]")
    print("  pip install engram[chroma]")
    print("  pip install engram[qdrant]")
    print("  pip install engram[redis]")
    print("  pip install engram[neo4j]")
    print("  pip install engram[mem0]")
    print("  pip install engram[zep]")


if __name__ == "__main__":
    asyncio.run(main())
