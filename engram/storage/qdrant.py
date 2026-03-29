from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("qdrant_client",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class QdrantDriver(InMemoryDriver):
    def __init__(self, collection_name: str = "engram", **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "QdrantDriver requires the optional `qdrant-client` dependency. "
                "Install the Qdrant extra before constructing this driver."
            )
        super().__init__()
        self.collection_name = collection_name
        self.options = dict(options)
