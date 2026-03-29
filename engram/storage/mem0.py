from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("mem0",)


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class Mem0Driver(InMemoryDriver):
    def __init__(self, **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "Mem0Driver requires the optional `mem0` dependency. "
                "Install the Mem0 extra before constructing this driver."
            )
        super().__init__()
        self.options = dict(options)
