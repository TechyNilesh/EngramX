"""
Background Lifecycle Jobs with PolicyJobScheduler
==================================================

This example demonstrates how to use ``engram.jobs.PolicyJobScheduler``
to run memory-lifecycle operations (decay, promotion, governance) as
background jobs.

Covered topics:

1. Creating a MemoryClient with extraction, retention, summarization,
   and governance policies
2. Adding memories at different simulated ages
3. Running a single lifecycle pass with ``PolicyJobScheduler.run_once()``
4. Inspecting the ``JobRunReport`` for decay, promotion, and deletion details
5. Starting and stopping the continuous scheduler (commented out)

No external dependencies are needed -- run this file directly:

    python examples/11_background_jobs.py
"""

import asyncio
from datetime import datetime, timedelta, timezone

from engramx.client import MemoryClient
from engramx.jobs import PolicyJobScheduler


# ---------------------------------------------------------------------------
# Helper -- build a fully-configured MemoryClient
# ---------------------------------------------------------------------------

def build_client() -> MemoryClient:
    """Return a MemoryClient with a rich set of lifecycle policies."""

    config = {
        "driver": {"kind": "memory"},
        "policies": {
            # Extraction: pull out user preferences from conversation turns.
            "extraction": [
                {
                    "name": "extract-preferences",
                    "trigger": "conversation_turn",
                    "conditions": [{"content_matches": ["prefer"]}],
                    "create": {
                        "type": "semantic",
                        "scope": "user",
                        "importance_score": 0.8,
                        "sensitivity_level": "internal",
                    },
                },
            ],
            # Retention: session memories decay exponentially with a 7-day
            # half-life; episodic memories decay more slowly.
            "retention": [
                {
                    "name": "decay-session",
                    "applies_to": {"scope": "session"},
                    "decay_function": "exponential",
                    "half_life_days": 7,
                    "floor_score": 0.1,
                },
                {
                    "name": "decay-episodic",
                    "applies_to": {"type": "episodic"},
                    "decay_function": "exponential",
                    "half_life_days": 30,
                    "floor_score": 0.2,
                },
            ],
            # Summarization: when two or more semantic memories share similar
            # content, promote them into a procedural memory.
            "summarization": [
                {
                    "name": "promote-to-procedural",
                    "applies_to": {"type": "semantic"},
                    "trigger": "same_content_count >= 2",
                    "promote_to": {"type": "procedural", "scope": "agent"},
                },
            ],
            # Governance: delete session-scoped memories older than 14 days;
            # audit every deletion.
            "governance": [
                {
                    "name": "purge-old-sessions",
                    "applies_to": {"scope": "session"},
                    "retention_days": 14,
                    "on_user_deletion": "delete",
                    "audit_log": True,
                },
            ],
        },
    }

    return MemoryClient(config=config)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

async def main() -> None:
    client = build_client()
    now = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 1: Populate memories at different ages
    # ------------------------------------------------------------------
    print("=== Adding memories at various ages ===\n")

    # Recent session memory (1 hour old)
    recent_id = await client.add(
        type="episodic",
        scope="session",
        content="User asked about the weather in Tokyo.",
        user_id="user-1",
        session_id="sess-recent",
        timestamp_created=now - timedelta(hours=1),
        timestamp_updated=now - timedelta(hours=1),
    )
    print(f"  [recent]  {recent_id[:12]}...  (1 hour old, session)")

    # Medium-age session memory (10 days old)
    medium_id = await client.add(
        type="episodic",
        scope="session",
        content="User discussed vacation plans.",
        user_id="user-1",
        session_id="sess-medium",
        timestamp_created=now - timedelta(days=10),
        timestamp_updated=now - timedelta(days=10),
    )
    print(f"  [medium]  {medium_id[:12]}...  (10 days old, session)")

    # Old session memory (20 days old -- exceeds 14-day governance limit)
    old_id = await client.add(
        type="episodic",
        scope="session",
        content="User requested a recipe for pasta.",
        user_id="user-1",
        session_id="sess-old",
        timestamp_created=now - timedelta(days=20),
        timestamp_updated=now - timedelta(days=20),
    )
    print(f"  [old]     {old_id[:12]}...  (20 days old, session)")

    # Two semantic memories with overlapping content (for promotion)
    sem1_id = await client.add(
        type="semantic",
        scope="user",
        content="User prefers dark mode across all applications.",
        user_id="user-1",
        importance_score=0.8,
    )
    sem2_id = await client.add(
        type="semantic",
        scope="user",
        content="User prefers dark mode in their IDE.",
        user_id="user-1",
        importance_score=0.8,
    )
    print(f"  [sem-1]   {sem1_id[:12]}...  (semantic, dark mode pref)")
    print(f"  [sem-2]   {sem2_id[:12]}...  (semantic, dark mode pref)")

    # ------------------------------------------------------------------
    # Step 2: Create the scheduler and run one lifecycle pass
    # ------------------------------------------------------------------
    print("\n=== Running PolicyJobScheduler.run_once() ===\n")

    scheduler = PolicyJobScheduler(
        driver=client.driver,
        policy_engine=client.policy_engine,
        interval_seconds=60.0,  # only relevant for the continuous loop
    )

    report = await scheduler.run_once(now=now)

    # ------------------------------------------------------------------
    # Step 3: Inspect the JobRunReport
    # ------------------------------------------------------------------
    print(f"Job started at:  {report.started_at.isoformat()}")
    print(f"Job finished at: {report.finished_at.isoformat()}")

    # Note: some decay-updated memories may have been subsequently deleted
    # by governance in the same pass, so we guard the get() call.
    print(f"\nDecay updated: {len(report.decay_updated_ids)} memory(ies)")
    for mid in report.decay_updated_ids:
        try:
            rec = await client.get(mid)
            print(f"  {mid[:12]}...  decay_factor={rec.decay_factor:.4f}  {rec.content[:50]}")
        except KeyError:
            print(f"  {mid[:12]}...  (subsequently deleted by governance)")

    print(f"\nPromoted: {len(report.promoted_ids)} memory(ies)")
    for mid in report.promoted_ids:
        try:
            rec = await client.get(mid)
            print(f"  {mid[:12]}...  type={rec.type}  derived_from={rec.derived_from}")
            print(f"    content: {rec.content[:70]}")
        except KeyError:
            print(f"  {mid[:12]}...  (subsequently deleted by governance)")

    print(f"\nDeleted (governance): {len(report.deleted_ids)} memory(ies)")
    for mid in report.deleted_ids:
        print(f"  {mid[:12]}...  (removed from store)")

    print(f"\nAudit log entries: {len(report.audit_log)}")
    for entry in report.audit_log:
        print(f"  rule={entry.rule_name}  action={entry.action}  "
              f"memory={entry.memory_id[:12]}...  reason={entry.reason}")

    # ------------------------------------------------------------------
    # Step 4: Verify what remains in the store
    # ------------------------------------------------------------------
    remaining = await client.list()
    print(f"\nRemaining memories after lifecycle pass: {len(remaining)}")
    for rec in remaining:
        print(f"  [{rec.type:>10}] scope={rec.scope:<8} "
              f"decay={rec.decay_factor:.4f}  {rec.content[:50]}")


# ---------------------------------------------------------------------------
# Continuous scheduler (commented -- runs forever until stopped)
# ---------------------------------------------------------------------------

async def continuous_scheduler_demo() -> None:
    """Start the scheduler and let it run every 30 seconds.

    Uncomment the call in ``__main__`` to try this interactively.
    Press Ctrl+C to stop.
    """
    client = build_client()

    # Add a memory so the scheduler has something to process.
    await client.add(
        type="episodic",
        scope="session",
        content="Background job test memory.",
        user_id="user-1",
        session_id="sess-bg",
    )

    scheduler = PolicyJobScheduler(
        driver=client.driver,
        policy_engine=client.policy_engine,
        interval_seconds=30.0,
    )

    # start() launches the loop as a background asyncio task.
    await scheduler.start()
    print("Scheduler started (runs every 30s). Press Ctrl+C to stop.\n")

    try:
        # Keep the event loop alive until interrupted.
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await scheduler.stop()
        print("Scheduler stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(main())

    # Uncomment the line below to try the continuous scheduler:
    # asyncio.run(continuous_scheduler_demo())
