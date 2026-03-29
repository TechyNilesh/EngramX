from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("chromadb",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class ChromaDriver(InMemoryDriver):
    def __init__(self, collection_name: str = "engram", **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "ChromaDriver requires the optional `chromadb` dependency. "
                "Install the Chroma extra before constructing this driver."
            )
        super().__init__()
        self.collection_name = collection_name
        self.options = dict(options)
