from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Mapping

from .config import EngramConfig, load_config
from .embedding import Embedder, HashEmbedder
from .models import MemorySignal
from .observability import MemoryTrace, OutputAttribution, ScoredMemory
from .policy import PolicyEngine, PolicyOutcome
from .schema import MemoryRecord, new_memory_id
from .storage.base import BaseDriver
from .storage.memory import InMemoryDriver
from .storage.sqlite import SQLiteDriver


def _run_sync(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Sync wrappers cannot run inside an active event loop")


class MemoryClient:
    def __init__(
        self,
        config: str | Mapping[str, Any] | EngramConfig | None = None,
        driver: BaseDriver | str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        if isinstance(config, BaseDriver):
            driver = config
            config = None
        elif isinstance(config, str) and config in {"memory", "sqlite"} and driver is None:
            driver = config
            config = None
        if isinstance(config, EngramConfig):
            self.config = config
        elif config is None:
            self.config = EngramConfig()
        else:
            self.config = load_config(config)
        self.driver = self._resolve_driver(driver or self.config.driver.kind)
        self.embedder = embedder or HashEmbedder()
        self.policy_engine = PolicyEngine(self.config.policies) if self.config.policies.has_rules() else None

    def _normalize_roles(self, access_roles: Any | None) -> list[str]:
        if access_roles is None:
            return []
        if isinstance(access_roles, str):
            return [access_roles]
        return [str(role) for role in access_roles]

    def _merge_filters(
        self,
        filters: dict[str, Any] | None,
        access_roles: Any | None,
    ) -> dict[str, Any]:
        resolved_filters = dict(filters or {})
        resolved_roles = self._normalize_roles(access_roles)
        if resolved_roles:
            resolved_filters.setdefault("access_roles", resolved_roles)
        return resolved_filters

    def _can_access(self, record: MemoryRecord, access_roles: Any | None) -> bool:
        if not record.access_roles:
            return True
        roles = set(self._normalize_roles(access_roles))
        if not roles:
            return False
        return bool(roles.intersection(record.access_roles))

    def _resolve_driver(self, driver: BaseDriver | str) -> BaseDriver:
        if isinstance(driver, BaseDriver):
            return driver
        if driver == "memory":
            return InMemoryDriver()
        if driver == "sqlite":
            return SQLiteDriver(self.config.driver.path)
        raise ValueError(f"Unsupported driver: {driver}")

    async def add(self, record: MemoryRecord | None = None, **kwargs: Any) -> str:
        if record is None:
            kwargs.setdefault("id", new_memory_id())
            record = MemoryRecord.create(**kwargs)
        return await self.driver.upsert(record)

    def add_sync(self, record: MemoryRecord | None = None, **kwargs: Any) -> str:
        return _run_sync(self.add(record=record, **kwargs))

    async def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        *,
        access_roles: Any | None = None,
    ) -> list[ScoredMemory]:
        resolved_filters = self._merge_filters(filters, access_roles)
        resolved_filters.setdefault("query_embedding", self.embedder.embed(query))
        resolved_filters.setdefault("mode", "hybrid")
        return await self.driver.search(query, resolved_filters, top_k)

    def search_sync(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        *,
        access_roles: Any | None = None,
    ) -> list[ScoredMemory]:
        return _run_sync(self.search(query=query, filters=filters, top_k=top_k, access_roles=access_roles))

    async def list(
        self,
        filters: dict[str, Any] | None = None,
        *,
        access_roles: Any | None = None,
    ) -> list[MemoryRecord]:
        return await self.driver.list(self._merge_filters(filters, access_roles))

    async def get(self, memory_id: str, *, access_roles: Any | None = None) -> MemoryRecord:
        record = await self.driver.get(memory_id)
        if not self._can_access(record, access_roles):
            raise PermissionError(f"Access denied for memory_id={memory_id}")
        return record

    async def update(self, memory_id: str, patch: dict[str, Any]) -> MemoryRecord:
        return await self.driver.update(memory_id, patch)

    async def delete(self, filters: dict[str, Any]) -> int:
        return await self.driver.delete(filters)

    async def explain(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        *,
        access_roles: Any | None = None,
    ) -> MemoryTrace:
        resolved_filters = self._merge_filters(filters, access_roles)
        resolved_filters.setdefault("query_embedding", self.embedder.embed(query))
        resolved_filters.setdefault("mode", "hybrid")
        return await self.driver.explain(query, resolved_filters, top_k)

    async def timeline(self, **kwargs: Any):
        return await self.driver.timeline(**kwargs)

    async def ingest_event(self, event: MemorySignal | Mapping[str, Any]) -> PolicyOutcome:
        if self.policy_engine is None:
            return PolicyOutcome()
        outcome = self.policy_engine.process_event(event)
        for record in outcome.extracted + outcome.promoted:
            await self.driver.upsert(record)
        return outcome

    async def promote(self) -> list[MemoryRecord]:
        if self.policy_engine is None:
            return []
        records = await self.driver.list()
        promoted = self.policy_engine.promote(records)
        existing = {tuple(sorted(record.derived_from)): record.id for record in records if record.type == "procedural"}
        created: list[MemoryRecord] = []
        for record in promoted:
            signature = tuple(sorted(record.derived_from))
            if signature in existing:
                continue
            await self.driver.upsert(record)
            created.append(record)
        return created

    async def apply_decay(self) -> list[MemoryRecord]:
        if self.policy_engine is None:
            return []
        changed: list[MemoryRecord] = []
        for record in await self.driver.list():
            decay = self.policy_engine.apply_decay(record)
            if abs(decay - record.decay_factor) < 1e-9:
                continue
            updated = await self.driver.update(record.id, {"decay_factor": decay})
            changed.append(updated)
        return changed

    async def forget_user(self, user_id: str) -> int:
        if self.policy_engine is None:
            return await self.driver.delete({"user_id": user_id})
        records = await self.driver.list({"user_id": user_id})
        kept, deleted = self.policy_engine.apply_governance(records, user_id=user_id)
        keep_ids = {record.id for record in kept}
        for record in records:
            if record.id not in keep_ids:
                await self.driver.delete({"id": record.id})
        return deleted

    def build_attribution(
        self,
        memories: list[ScoredMemory],
        policies_fired: list[str] | None = None,
    ) -> OutputAttribution:
        return OutputAttribution(
            memories_used=[item.record.id for item in memories],
            policies_fired=list(policies_fired or []),
        )

    async def run_with_attribution(
        self,
        task: str,
        *,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        access_roles: Any | None = None,
        runner: Callable[[str, list[ScoredMemory]], Awaitable[str] | str] | None = None,
    ) -> tuple[str, OutputAttribution]:
        resolved_filters = self._merge_filters(filters, access_roles)
        if user_id is not None:
            resolved_filters.setdefault("user_id", user_id)
        memories = await self.search(task, filters=resolved_filters, top_k=top_k, access_roles=access_roles)
        policies_fired: list[str] = []
        if self.policy_engine is not None:
            outcome = self.policy_engine.process_event(
                MemorySignal(
                    type="conversation_turn",
                    content=task,
                    source="conversation",
                    user_id=user_id,
                    metadata={"task": True},
                )
            )
            policies_fired = list(outcome.policies_fired)
        memory_context = "\n".join(f"- {item.record.content}" for item in memories) or "- None"
        policy_context = "\n".join(f"- {name}" for name in policies_fired) or "- None"
        prompt = f"Task: {task}\nRelevant memories:\n{memory_context}\nPolicies fired:\n{policy_context}"
        if runner is None:
            response = prompt
        else:
            result = runner(prompt, memories)
            if asyncio.iscoroutine(result):
                response = await result
            else:
                response = str(result)
        attribution = self.build_attribution(
            memories=memories,
            policies_fired=policies_fired,
        )
        return response, attribution
