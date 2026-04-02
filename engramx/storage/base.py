from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..observability import MemoryTimelineEvent, MemoryTrace, ScoredMemory
from ..schema import MemoryRecord


class BaseDriver(ABC):
    @abstractmethod
    async def upsert(self, record: MemoryRecord) -> str: ...

    @abstractmethod
    async def search(self, query: str, filters: dict[str, Any], top_k: int) -> list[ScoredMemory]: ...

    @abstractmethod
    async def list(self, filters: dict[str, Any] | None = None) -> list[MemoryRecord]: ...

    @abstractmethod
    async def get(self, memory_id: str) -> MemoryRecord: ...

    @abstractmethod
    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord: ...

    @abstractmethod
    async def delete(self, filters: dict[str, Any]) -> int: ...

    @abstractmethod
    async def timeline(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        from_dt: str | None = None,
        to_dt: str | None = None,
        types: list[str] | None = None,
    ) -> list[MemoryTimelineEvent]: ...

    @abstractmethod
    async def explain(self, query: str, filters: dict[str, Any], top_k: int = 5) -> MemoryTrace: ...
