from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from engramxx.config import load_policy_config
from engramxx.models import MemoryEvent, MemoryRecord
from engramxx.policy import PolicyEngine


class PolicyEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)

    def test_extraction_creates_semantic_memory(self) -> None:
        config = load_policy_config(
            {
                "policies": {
                    "extraction": [
                        {
                            "name": "extract-user-preferences",
                            "trigger": "conversation_turn",
                            "conditions": [{"content_matches": ["prefer", "metric"]}],
                            "create": {
                                "type": "semantic",
                                "scope": "user",
                                "importance_score": 0.8,
                                "sensitivity_level": "internal",
                            },
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        outcome = engine.process_event(
            MemoryEvent(
                type="conversation_turn",
                content="I always prefer metric units for measurements.",
                user_id="u-123",
                session_id="s-1",
            ),
            now=self.now,
        )

        self.assertEqual(1, len(outcome.extracted))
        record = outcome.extracted[0]
        self.assertEqual("semantic", record.type)
        self.assertEqual("user", record.scope)
        self.assertEqual("u-123", record.user_id)
        self.assertEqual("internal", record.sensitivity_level)

    def test_yaml_config_loads_from_string(self) -> None:
        config = load_policy_config(
            """
            policies:
              extraction:
                - name: extract-user-preferences
                  trigger: conversation_turn
                  conditions:
                    - content_matches: [prefer, metric]
                  create:
                    type: semantic
                    scope: user
                    importance_score: 0.8
            """
        )

        self.assertEqual(1, len(config.extraction))
        self.assertEqual("extract-user-preferences", config.extraction[0].name)

    def test_importance_score_formula_uses_retry_count(self) -> None:
        config = load_policy_config(
            {
                "policies": {
                    "extraction": [
                        {
                            "name": "log-all-tool-failures",
                            "trigger": "tool_call_error",
                            "create": {
                                "type": "episodic",
                                "scope": "session",
                                "importance_score_formula": "1.0 - (0.1 * retry_count)",
                                "sensitivity_level": "internal",
                            },
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        outcome = engine.process_event(
            MemoryEvent(
                type="tool_call_error",
                content="Payment API returned 402.",
                retry_count=3,
                session_id="s-1",
            ),
            now=self.now,
        )

        self.assertEqual(1, len(outcome.extracted))
        self.assertAlmostEqual(0.7, outcome.extracted[0].importance_score, places=2)

    def test_apply_retention_updates_decay_factor(self) -> None:
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
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        record = MemoryRecord(
            id="mem-1",
            type="episodic",
            scope="session",
            content="Tool call failed.",
            timestamp_created=self.now - timedelta(days=7),
            timestamp_updated=self.now - timedelta(days=7),
        )

        updated = engine.apply_retention([record], now=self.now)
        self.assertAlmostEqual(0.5, updated[0].decay_factor, places=2)

    def test_decay_uses_exponential_half_life(self) -> None:
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
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        record = MemoryRecord(
            id="mem-1",
            type="episodic",
            scope="session",
            content="Tool call failed.",
            timestamp_created=self.now - timedelta(days=7),
            timestamp_updated=self.now - timedelta(days=7),
        )

        decay = engine.apply_decay(record, now=self.now)
        self.assertAlmostEqual(0.5, decay, places=2)

    def test_governance_deletes_user_sensitive_records(self) -> None:
        config = load_policy_config(
            {
                "policies": {
                    "governance": [
                        {
                            "name": "gdpr-user-data",
                            "applies_to": {
                                "scope": "user",
                                "sensitivity_level": ["confidential", "restricted"],
                            },
                            "retention_days": 30,
                            "on_user_deletion": "delete_all",
                            "audit_log": True,
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        records = [
            MemoryRecord(
                id="mem-1",
                type="semantic",
                scope="user",
                content="Sensitive preference",
                user_id="u-123",
                sensitivity_level="confidential",
            ),
            MemoryRecord(
                id="mem-2",
                type="semantic",
                scope="user",
                content="Another sensitive preference",
                user_id="u-123",
                sensitivity_level="restricted",
            ),
            MemoryRecord(
                id="mem-3",
                type="semantic",
                scope="user",
                content="Public note",
                user_id="u-123",
                sensitivity_level="public",
            ),
        ]

        kept, deleted = engine.apply_governance(records, user_id="u-123", now=self.now)
        self.assertEqual(2, deleted)
        self.assertEqual(["mem-3"], [record.id for record in kept])

    def test_governance_can_return_audit_log(self) -> None:
        config = load_policy_config(
            {
                "policies": {
                    "governance": [
                        {
                            "name": "gdpr-user-data",
                            "applies_to": {
                                "scope": "user",
                                "sensitivity_level": ["confidential", "restricted"],
                            },
                            "retention_days": 30,
                            "on_user_deletion": "delete_all",
                            "audit_log": True,
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        records = [
            MemoryRecord(
                id="mem-1",
                type="semantic",
                scope="user",
                content="Sensitive preference",
                user_id="u-123",
                sensitivity_level="confidential",
            )
        ]

        kept, deleted, audit_log = engine.apply_governance(
            records,
            user_id="u-123",
            now=self.now,
            include_audit_log=True,
        )
        self.assertEqual([], kept)
        self.assertEqual(1, deleted)
        self.assertEqual(1, len(audit_log))
        self.assertEqual("delete", audit_log[0].action)
        self.assertEqual("gdpr-user-data", audit_log[0].rule_name)

    def test_governance_retention_period_expires_records(self) -> None:
        config = load_policy_config(
            {
                "policies": {
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
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        records = [
            MemoryRecord(
                id="mem-old",
                type="semantic",
                scope="user",
                content="Old sensitive preference",
                user_id="u-123",
                sensitivity_level="confidential",
                timestamp_created=self.now - timedelta(days=31),
                timestamp_updated=self.now - timedelta(days=31),
            )
        ]

        kept, deleted = engine.apply_governance(records, reason="retention", now=self.now)
        self.assertEqual(1, deleted)
        self.assertEqual([], kept)

    def test_semantic_distillation_uses_reflector_and_summary_model(self) -> None:
        def reflector(records, rule, now):
            return {
                "content": "User prefers metric units.",
                "summary_model": rule.summary_model,
                "source_record_ids": [record.id for record in records],
                "generated_at": now.isoformat(),
            }

        config = load_policy_config(
            {
                "policies": {
                    "summarization": [
                        {
                            "name": "distill-user-preference",
                            "applies_to": {"type": "episodic", "scope": "session"},
                            "trigger": "same_action_success_count >= 2",
                            "promote_to": {
                                "type": "semantic",
                                "scope": "user",
                            },
                            "summary_model": "gpt-4o-mini",
                            "reflector": reflector,
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        episodes = [
            MemoryRecord(
                id=f"ep-{index}",
                type="episodic",
                scope="session",
                content="User prefers metric units.",
                user_id="u-123",
                payload={"action_signature": "unit_preference", "success": True},
                importance_score=0.8,
            )
            for index in range(2)
        ]

        summaries = engine.summarize(episodes, now=self.now)
        self.assertEqual(1, len(summaries))
        summary = summaries[0]
        self.assertEqual("semantic", summary.type)
        self.assertEqual("user", summary.scope)
        self.assertEqual("User prefers metric units.", summary.content)
        self.assertEqual("gpt-4o-mini", summary.payload["summary_model"])
        self.assertEqual(["ep-0", "ep-1"], summary.payload["source_record_ids"])

    def test_promotion_creates_procedural_memory_after_repeated_success(self) -> None:
        config = load_policy_config(
            {
                "policies": {
                    "summarization": [
                        {
                            "name": "promote-repeated-success",
                            "applies_to": {"type": "episodic"},
                            "trigger": "same_action_success_count >= 3",
                            "promote_to": {
                                "type": "procedural",
                                "scope": "agent",
                                "content": "Check visa requirements before searching flights.",
                            },
                        }
                    ]
                }
            }
        )
        engine = PolicyEngine(config)
        episodes = [
            MemoryRecord(
                id=f"ep-{index}",
                type="episodic",
                scope="session",
                content="Travel booking succeeded.",
                user_id="u-123",
                payload={
                    "action_signature": "travel_booking",
                    "success": True,
                    "action_template": "check visa before flights",
                },
                importance_score=0.7,
            )
            for index in range(3)
        ]

        promoted = engine.promote(episodes, now=self.now)
        self.assertEqual(1, len(promoted))
        self.assertEqual("procedural", promoted[0].type)
        self.assertEqual("agent", promoted[0].scope)
        self.assertIn("ep-0", promoted[0].derived_from)


if __name__ == "__main__":
    unittest.main()
