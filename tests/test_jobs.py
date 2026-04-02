from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from engramxx.config import load_policy_config
from engramxx.jobs import PolicyJobScheduler
from engramxx.policy import PolicyEngine
from engramxx.schema import MemoryRecord
from engramxx.storage.memory import InMemoryDriver


@pytest.mark.asyncio
async def test_scheduler_runs_decay_promotion_and_governance():
    now = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)
    config = load_policy_config(
        {
            "policies": {
                "retention": [
                    {
                        "name": "session-episode-decay",
                        "applies_to": {"type": "episodic", "scope": "session"},
                        "decay_function": "exponential",
                        "half_life_days": 7,
                        "floor_score": 0.1,
                    }
                ],
                "summarization": [
                    {
                        "name": "distill-user-preference",
                        "applies_to": {"type": "episodic", "scope": "session"},
                        "trigger": "same_action_success_count >= 2",
                        "promote_to": {"type": "semantic", "scope": "user"},
                        "summary_model": "gpt-4o-mini",
                    }
                ],
                "governance": [
                    {
                        "name": "gdpr-user-data",
                        "applies_to": {
                            "scope": "user",
                            "sensitivity_level": ["confidential", "restricted"],
                        },
                        "retention_days": 30,
                        "on_user_deletion": "retain",
                        "audit_log": True,
                    }
                ],
            }
        }
    )
    engine = PolicyEngine(config)
    driver = InMemoryDriver()

    await driver.upsert(
        MemoryRecord(
            id="ep-old",
            type="episodic",
            scope="session",
            user_id="u-123",
            content="Background note to decay.",
            importance_score=0.5,
            decay_factor=1.0,
            timestamp_created=now - timedelta(days=7),
            timestamp_updated=now - timedelta(days=7),
        )
    )
    await driver.upsert(
        MemoryRecord(
            id="ep-1",
            type="episodic",
            scope="session",
            user_id="u-123",
            content="User prefers metric units.",
            payload={"action_signature": "unit-preference", "success": True},
            importance_score=0.8,
        )
    )
    await driver.upsert(
        MemoryRecord(
            id="ep-2",
            type="episodic",
            scope="session",
            user_id="u-123",
            content="User prefers metric units.",
            payload={"action_signature": "unit-preference", "success": True},
            importance_score=0.8,
        )
    )
    await driver.upsert(
        MemoryRecord(
            id="confidential-old",
            type="semantic",
            scope="user",
            user_id="u-123",
            content="Sensitive preference",
            sensitivity_level="confidential",
            timestamp_created=now - timedelta(days=31),
            timestamp_updated=now - timedelta(days=31),
        )
    )

    scheduler = PolicyJobScheduler(driver, engine, interval_seconds=0.01)
    report = await scheduler.run_once(now=now)

    assert "ep-old" in report.decay_updated_ids
    assert len(report.promoted_ids) == 1
    assert len(report.deleted_ids) == 1
    assert report.audit_log

    remaining = await driver.list()
    remaining_ids = {record.id for record in remaining}
    assert "confidential-old" not in remaining_ids
    assert any(record.type == "semantic" and record.id in report.promoted_ids for record in remaining)
