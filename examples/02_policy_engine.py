"""
02_policy_engine.py -- Engram Policy Engine walkthrough

Demonstrates the full policy lifecycle using the in-memory driver:

1. Define extraction, retention, summarization, and governance rules in YAML.
2. Ingest conversation events so the extraction rules create episodic memories.
3. Run exponential decay to age memories over simulated time.
4. Promote repeated patterns into procedural memory.
5. Exercise GDPR-style user deletion via governance rules.

Each step prints its results so you can follow along in the terminal.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from engramx.client import MemoryClient
from engramx.models import MemorySignal

# ---------------------------------------------------------------------------
# 1. Policy configuration expressed as a YAML string
# ---------------------------------------------------------------------------

POLICY_YAML = """
policies:
  # -- Extraction rules ------------------------------------------------
  # When a conversation_turn arrives whose content mentions "prefer",
  # create an episodic, user-scoped memory tagged as a preference.
  extraction:
    - name: capture_user_preferences
      trigger: conversation_turn
      conditions:
        - content_matches: ["prefer"]
      create:
        type: episodic
        scope: user
        sensitivity_level: internal
        payload:
          tag: user_preference

    # A second rule captures any conversation turn that contains
    # "confidential" and marks the memory accordingly.
    - name: capture_confidential_notes
      trigger: conversation_turn
      conditions:
        - content_matches: ["confidential"]
      create:
        type: episodic
        scope: user
        sensitivity_level: confidential
        payload:
          tag: confidential_note

  # -- Retention rules -------------------------------------------------
  # Episodic memories decay exponentially with a 7-day half-life.
  # The floor_score ensures memories never drop below 0.1.
  retention:
    - name: short_term_decay
      applies_to:
        type: episodic
      decay_function: exponential
      half_life_days: 7
      floor_score: 0.1

  # -- Summarization rules ---------------------------------------------
  # After 3 episodic memories with the same normalised content appear,
  # promote them into a single procedural memory.
  summarization:
    - name: promote_repeated_preference
      applies_to:
        type: episodic
      trigger: "same_content_count >= 3"
      promote_to:
        type: procedural
        scope: agent

  # -- Governance rules ------------------------------------------------
  # Confidential memories are deleted outright on user-deletion request
  # (GDPR "right to erasure").  An audit trail is kept.
  governance:
    - name: gdpr_confidential_delete
      applies_to:
        sensitivity_level: confidential
      on_user_deletion: delete_all
      audit_log: true

    # Non-confidential data is retained (default behaviour) so the
    # agent can keep anonymised procedural knowledge.
    - name: retain_general_knowledge
      applies_to:
        sensitivity_level: internal
      on_user_deletion: retain
      audit_log: true
"""


# ---------------------------------------------------------------------------
# Helper to print a divider
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def print_records(records, label: str = "Records") -> None:
    if not records:
        print(f"  ({label}: none)")
        return
    for rec in records:
        decay = f", decay={rec.decay_factor:.4f}" if hasattr(rec, "decay_factor") else ""
        sensitivity = f", sensitivity={rec.sensitivity_level}" if hasattr(rec, "sensitivity_level") else ""
        print(
            f"  [{rec.type:>10}] {rec.content[:60]:<60} "
            f"(id=...{rec.id[-6:]}{decay}{sensitivity})"
        )


# ---------------------------------------------------------------------------
# Main async workflow
# ---------------------------------------------------------------------------

async def main() -> None:
    # ------------------------------------------------------------------
    # 2. Load the YAML config and create a MemoryClient (in-memory driver)
    # ------------------------------------------------------------------
    section("Step 1: Create MemoryClient from YAML policy config")
    client = MemoryClient(POLICY_YAML)

    print(f"  Driver  : {type(client.driver).__name__}")
    print(f"  Policies: extraction={len(client.config.policies.extraction)}, "
          f"retention={len(client.config.policies.retention)}, "
          f"summarization={len(client.config.policies.summarization)}, "
          f"governance={len(client.config.policies.governance)}")

    # ------------------------------------------------------------------
    # 3. Ingest conversation events
    # ------------------------------------------------------------------
    section("Step 2: Ingest conversation events")

    user_id = "user-42"

    # Three nearly-identical preference signals (will trigger promotion later).
    preference_events = [
        MemorySignal(
            type="conversation_turn",
            content="I prefer dark mode for all my editors",
            source="chat",
            user_id=user_id,
        ),
        MemorySignal(
            type="conversation_turn",
            content="I prefer dark mode for all my editors",
            source="chat",
            user_id=user_id,
        ),
        MemorySignal(
            type="conversation_turn",
            content="I prefer dark mode for all my editors",
            source="chat",
            user_id=user_id,
        ),
    ]

    # One confidential event (will be removed by governance later).
    confidential_event = MemorySignal(
        type="conversation_turn",
        content="This is confidential: my API key is sk-secret-123",
        source="chat",
        user_id=user_id,
    )

    for i, event in enumerate(preference_events, 1):
        outcome = await client.ingest_event(event)
        print(f"  Event {i}: extracted={len(outcome.extracted)}, "
              f"policies_fired={outcome.policies_fired}")

    outcome = await client.ingest_event(confidential_event)
    print(f"  Event 4 (confidential): extracted={len(outcome.extracted)}, "
          f"policies_fired={outcome.policies_fired}")

    # Show all stored memories so far.
    all_records = await client.list()
    print(f"\n  Total memories in store: {len(all_records)}")
    print_records(all_records, "All memories")

    # ------------------------------------------------------------------
    # 4a. Apply exponential decay
    # ------------------------------------------------------------------
    section("Step 3: Apply retention decay (simulating 10 days later)")

    # To observe meaningful decay we patch timestamps to look 10 days old.
    ten_days_ago = datetime.now(timezone.utc) - timedelta(days=10)
    for record in all_records:
        await client.update(record.id, {
            "timestamp_created": ten_days_ago,
            "timestamp_updated": ten_days_ago,
        })

    decayed = await client.apply_decay()
    print(f"  Records with updated decay factor: {len(decayed)}")
    print_records(decayed, "Decayed memories")

    # With a 7-day half-life and 10 days elapsed, the expected factor is
    # 0.5^(10/7) ~ 0.37, floored at 0.1.
    if decayed:
        print(f"\n  (Expected decay ~ 0.5^(10/7) = {0.5 ** (10 / 7):.4f})")

    # ------------------------------------------------------------------
    # 4b. Promote repeated patterns to procedural memory
    # ------------------------------------------------------------------
    section("Step 4: Promote repeated patterns")

    promoted = await client.promote()
    print(f"  Newly promoted memories: {len(promoted)}")
    print_records(promoted, "Promoted")

    all_records = await client.list()
    print(f"\n  Total memories after promotion: {len(all_records)}")
    print_records(all_records, "All memories")

    # ------------------------------------------------------------------
    # 5. Governance -- GDPR forget_user
    # ------------------------------------------------------------------
    section("Step 5: GDPR forget_user (delete confidential data)")

    deleted_count = await client.forget_user(user_id)
    print(f"  Memories deleted by governance: {deleted_count}")

    remaining = await client.list()
    print(f"  Memories remaining: {len(remaining)}")
    print_records(remaining, "Remaining")

    # Show the audit trail recorded by the policy engine.
    if client.policy_engine and client.policy_engine.audit_events:
        print(f"\n  Audit log ({len(client.policy_engine.audit_events)} entries):")
        for entry in client.policy_engine.audit_events:
            print(f"    rule={entry.rule_name}, action={entry.action}, "
                  f"memory=...{entry.memory_id[-6:]}, reason={entry.reason}")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    section("Done")
    print("  The policy engine managed the full lifecycle:")
    print("    - Extracted preferences and confidential notes from conversation turns")
    print("    - Applied exponential decay (half_life=7 days) to episodic memories")
    print("    - Promoted repeated preferences into procedural memory")
    print("    - Deleted confidential data on GDPR user-deletion request")
    print()


if __name__ == "__main__":
    asyncio.run(main())
