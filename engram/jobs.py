from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .lifecycle import GovernanceAuditEntry
from .policy import PolicyEngine
from .schema import MemoryRecord, utcnow
from .storage.base import BaseDriver


@dataclass(slots=True)
class JobRunReport:
    started_at: datetime
    finished_at: datetime
    decay_updated_ids: list[str] = field(default_factory=list)
    promoted_ids: list[str] = field(default_factory=list)
    deleted_ids: list[str] = field(default_factory=list)
    audit_log: list[GovernanceAuditEntry] = field(default_factory=list)


class PolicyJobScheduler:
    def __init__(
        self,
        driver: BaseDriver,
        policy_engine: PolicyEngine,
        *,
        interval_seconds: float = 60.0,
    ) -> None:
        self.driver = driver
        self.policy_engine = policy_engine
        self.interval_seconds = interval_seconds
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def run_once(self, *, now: datetime | None = None) -> JobRunReport:
        started_at = now or utcnow()
        records = await self.driver.list()

        decay_updated_ids: list[str] = []
        for record, updated in zip(records, self.policy_engine.apply_retention(records, now=now)):
            if updated.decay_factor == record.decay_factor and updated.timestamp_updated == record.timestamp_updated:
                continue
            await self.driver.update(
                record.id,
                {
                    "decay_factor": updated.decay_factor,
                    "timestamp_updated": updated.timestamp_updated.isoformat(),
                },
            )
            decay_updated_ids.append(record.id)

        promoted_ids: list[str] = []
        existing_summaries = {
            tuple(sorted(record.derived_from))
            for record in records
            if record.type in {"semantic", "procedural", "meta"}
        }
        for summary in self.policy_engine.summarize(records, now=now):
            signature = tuple(sorted(summary.derived_from))
            if signature in existing_summaries:
                continue
            await self.driver.upsert(summary)
            promoted_ids.append(summary.id)
            existing_summaries.add(signature)

        kept, _, audit_log = self.policy_engine.apply_governance(
            records,
            now=now,
            include_audit_log=True,
        )
        kept_ids = {record.id for record in kept}
        deleted_ids: list[str] = []
        for record in records:
            if record.id in kept_ids:
                continue
            await self.driver.delete({"id": record.id})
            deleted_ids.append(record.id)

        finished_at = utcnow()
        return JobRunReport(
            started_at=started_at,
            finished_at=finished_at,
            decay_updated_ids=decay_updated_ids,
            promoted_ids=promoted_ids,
            deleted_ids=deleted_ids,
            audit_log=audit_log,
        )

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is None:
            return
        await self._task
        self._task = None

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue
