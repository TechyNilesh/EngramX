from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "class_name", "kwargs", "message"),
    [
        ("engram.storage.postgres", "PostgresDriver", {"dsn": "postgresql://example/db"}, "asyncpg"),
        ("engram.storage.chroma", "ChromaDriver", {"collection_name": "engram"}, "chromadb"),
        ("engram.storage.qdrant", "QdrantDriver", {"collection_name": "engram"}, "qdrant-client"),
        ("engram.storage.redis", "RedisDriver", {"url": "redis://localhost"}, "redis"),
        ("engram.storage.neo4j", "Neo4jDriver", {"uri": "bolt://localhost:7687"}, "neo4j"),
        ("engram.storage.mem0", "Mem0Driver", {}, "mem0"),
        ("engram.storage.zep", "ZepDriver", {"api_key": "test-key"}, "zep"),
    ],
)
def test_optional_driver_constructors_require_dependencies(monkeypatch, module_name, class_name, kwargs, message):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "_dependency_available", lambda: False)
    driver_cls = getattr(module, class_name)

    with pytest.raises(RuntimeError, match="(?i)" + message):
        driver_cls(**kwargs)
