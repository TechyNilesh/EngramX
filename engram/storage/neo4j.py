from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("neo4j",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class Neo4jDriver(InMemoryDriver):
    def __init__(self, uri: str, **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "Neo4jDriver requires the optional `neo4j` dependency. "
                "Install the Neo4j extra before constructing this driver."
            )
        super().__init__()
        self.uri = uri
        self.options = dict(options)
