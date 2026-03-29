from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("zep_python", "zep")


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class ZepDriver(InMemoryDriver):
    def __init__(self, api_key: str | None = None, **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "ZepDriver requires the optional Zep client dependency. "
                "Install the Zep extra before constructing this driver."
            )
        super().__init__()
        self.api_key = api_key
        self.options = dict(options)
