from __future__ import annotations

import importlib.util
from typing import Any

from .memory import InMemoryDriver

_DEPENDENCIES = ("psycopg", "psycopg2")


def _dependency_available() -> bool:
    return any(importlib.util.find_spec(name) is not None for name in _DEPENDENCIES)


class PostgresDriver(InMemoryDriver):
    def __init__(self, dsn: str, **options: Any) -> None:
        if not _dependency_available():
            raise RuntimeError(
                "PostgresDriver requires an installed PostgreSQL client dependency "
                f"({', '.join(_DEPENDENCIES)}). Install the optional extras for Engram."
            )
        super().__init__()
        self.dsn = dsn
        self.options = dict(options)
