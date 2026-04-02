from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .schema import (
    MemoryRecord,
    MemoryScope,
    MemoryType,
    SensitivityLevel,
    new_memory_id,
    parse_datetime,
    utcnow,
)


@dataclass(slots=True)
class MemorySignal:
    type: str
    content: str
    source: str = "conversation"
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utcnow)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "MemorySignal":
        payload = dict(data)
        if "timestamp" in payload:
            payload["timestamp"] = parse_datetime(payload["timestamp"])
        return cls(**payload)


MemoryEvent = MemorySignal

__all__ = [
    "MemoryEvent",
    "MemoryRecord",
    "MemoryScope",
    "MemorySignal",
    "MemoryType",
    "SensitivityLevel",
    "new_memory_id",
    "utcnow",
]
