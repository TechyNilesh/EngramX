from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("redis",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class RedisDriver(InMemoryDriver):
    def __init__(self, namespace: str = "engram", **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "RedisDriver requires the optional `redis` dependency. "
                "Install the Redis extra before constructing this driver."
            )
        super().__init__()
        self.namespace = namespace
        self.options = dict(options)
