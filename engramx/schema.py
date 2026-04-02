from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Literal
import random
import uuid

MemoryType = Literal["episodic", "semantic", "procedural", "meta"]
MemoryScope = Literal["user", "session", "agent", "global"]
SensitivityLevel = Literal["public", "internal", "confidential", "restricted"]

_DATETIME_FIELDS = (
    "timestamp_created",
    "timestamp_updated",
    "valid_from",
    "valid_to",
    "last_accessed",
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def new_memory_id() -> str:
    # Prefer native UUIDv7 when available (Python 3.14+), then fallback to a compatible
    # time-ordered UUID layout so IDs remain sortable by creation time.
    if hasattr(uuid, "uuid7"):
        return str(uuid.uuid7())  # type: ignore[attr-defined]
    ts_ms = int(utcnow().timestamp() * 1000) & ((1 << 48) - 1)
    rand_a = random.getrandbits(12)
    rand_b = random.getrandbits(62)
    value = 0
    value |= ts_ms << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b
    return str(uuid.UUID(int=value))


@dataclass(slots=True)
class MemoryRecord:
    id: str
    type: MemoryType
    scope: MemoryScope
    agent_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    content: str = ""
    payload: dict[str, Any] | None = None
    timestamp_created: datetime = field(default_factory=utcnow)
    timestamp_updated: datetime = field(default_factory=utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source: str = "conversation"
    provenance: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    importance_score: float = 0.5
    access_count: int = 0
    last_accessed: datetime | None = None
    decay_factor: float = 1.0
    sensitivity_level: SensitivityLevel = "public"
    retention_policy: str | None = None
    access_roles: list[str] = field(default_factory=list)
    embedding: list[float] | None = field(default=None, repr=False)
    embedding_model: str | None = None

    @classmethod
    def create(cls, **kwargs: Any) -> "MemoryRecord":
        payload = dict(kwargs)
        payload.setdefault("id", new_memory_id())
        payload.setdefault("timestamp_created", utcnow())
        payload.setdefault("timestamp_updated", payload["timestamp_created"])
        return cls(**payload)

    def touch(self, *, timestamp: datetime | None = None) -> "MemoryRecord":
        return replace(self, timestamp_updated=parse_datetime(timestamp) or utcnow())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in _DATETIME_FIELDS:
            value = data.get(key)
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        payload = dict(data)
        for key in _DATETIME_FIELDS:
            if key in payload:
                payload[key] = parse_datetime(payload.get(key))
        return cls(**payload)
