"""
LLM-Powered Reflection for Memory Summarization
=================================================

This example demonstrates how Engram uses LLM reflectors to distill
episodic memories into higher-level semantic or procedural knowledge.

The reflection pipeline works as follows:

    episodic records --> reflector (LLM or custom) --> promoted memory

Engram ships with three provider-backed reflectors (OpenAI, Anthropic,
LiteLLM) and a simple subclass hook so you can bring your own logic.

What this example covers:

1. Creating provider-backed reflectors (commented -- they need API keys)
2. Building a custom mock reflector that needs no external dependencies
3. Wiring the reflector into the PolicyEngine via summarization rules
4. Running the full flow: ingest episodic events, trigger summarization,
   and inspect the promoted semantic/procedural memories

No external dependencies are needed -- the in-memory driver and mock
reflector ship with this file.  Run directly:

    python examples/08_llm_reflection.py
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from engramx import MemoryClient
from engramx.config import (
    EngramConfig,
    ExtractionRule,
    PolicyConfig,
    SummarizationRule,
)
from engramx.reflection import LLMReflector, _build_prompt, _parse_llm_response
from engramx.schema import MemoryRecord


# =====================================================================
# 1. Provider-backed reflectors (require API keys -- shown for reference)
# =====================================================================

# --- OpenAI (default) ------------------------------------------------
# from engramx.reflection import create_reflector
# openai_reflector = create_reflector("openai", model="gpt-4o-mini")

# --- Anthropic -------------------------------------------------------
# anthropic_reflector = create_reflector("anthropic", model="claude-sonnet-4-20250514")

# --- LiteLLM (100+ providers) ----------------------------------------
# litellm_reflector = create_reflector("litellm", model="gpt-4o-mini")

# --- OpenAI-compatible endpoint (e.g. Ollama, Together, vLLM) --------
# from engramx.reflection import OpenAIReflector
# local_reflector = OpenAIReflector(
#     model="llama3",
#     base_url="http://localhost:11434/v1",
#     api_key="not-needed",
# )


# =====================================================================
# 2. Custom mock reflector (no external deps)
# =====================================================================

class MockReflector(LLMReflector):
    """A reflector that summarises records locally without an LLM.

    It extracts unique keywords from the episodic records and builds a
    short summary string.  This is useful for testing, CI pipelines, or
    anywhere you want deterministic output.
    """

    def __init__(self) -> None:
        # provider/model are cosmetic -- used in the returned metadata
        super().__init__(provider="mock", model="keyword-extractor")

    def _call_llm(self, prompt: str) -> str:
        """Instead of calling an LLM, extract keywords and return JSON."""
        # Pull out the lines between the record markers
        lines: list[str] = []
        in_block = False
        for line in prompt.splitlines():
            if line.startswith("--- Episodic Records ---"):
                in_block = True
                continue
            if line.startswith("--- End Records ---"):
                break
            if in_block and line.strip():
                # Strip the leading number/timestamp prefix
                text = line.split("]", 1)[-1].strip() if "]" in line else line.strip()
                lines.append(text)

        # Build a keyword-based summary
        words: dict[str, int] = {}
        for text in lines:
            for word in text.lower().split():
                cleaned = word.strip(".,!?;:'\"()[]")
                if len(cleaned) > 3:
                    words[cleaned] = words.get(cleaned, 0) + 1

        top_words = sorted(words, key=words.get, reverse=True)[:8]
        summary = (
            f"Distilled {len(lines)} episodes. "
            f"Key themes: {', '.join(top_words)}."
        )
        # Return valid JSON as the LLM would
        return json.dumps({"content": summary})


# =====================================================================
# 3. Using the reflector directly (outside the policy engine)
# =====================================================================

def standalone_reflector_demo() -> None:
    """Call a reflector directly on a list of MemoryRecords."""
    print("=" * 60)
    print("Standalone reflector demo")
    print("=" * 60)

    # Create a few episodic records by hand
    records = [
        MemoryRecord.create(
            type="episodic",
            scope="user",
            content="Deployed v2.3 of the billing service to production.",
            user_id="alice",
        ),
        MemoryRecord.create(
            type="episodic",
            scope="user",
            content="Rolled back billing service v2.3 due to payment timeout errors.",
            user_id="alice",
        ),
        MemoryRecord.create(
            type="episodic",
            scope="user",
            content="Fixed timeout by increasing connection pool; redeployed billing v2.3.1.",
            user_id="alice",
        ),
    ]

    reflector = MockReflector()

    # __call__ accepts records, an optional rule, and an optional timestamp
    result = reflector(records, rule=None, now=datetime.now(timezone.utc))

    print(f"\nReflector result:")
    print(f"  content       : {result['content']}")
    print(f"  provider      : {result['provider']}")
    print(f"  summary_model : {result['summary_model']}")
    print(f"  source_count  : {result['source_count']}")
    print(f"  source_ids    : {result['source_ids']}")

    # You can also inspect the prompt that would be sent to the LLM
    prompt = _build_prompt(records, rule=None, now=datetime.now(timezone.utc))
    print(f"\nGenerated prompt (first 200 chars):\n  {prompt[:200]}...")

    # And parse raw LLM output manually
    raw = '{"content": "hello world"}'
    parsed = _parse_llm_response(raw)
    print(f"\nParsed LLM response: {parsed!r}")
    print()


# =====================================================================
# 4. Full pipeline: episodic events --> reflector --> promoted memory
# =====================================================================

async def policy_pipeline_demo() -> None:
    """Wire the mock reflector into the policy engine and run end-to-end."""
    print("=" * 60)
    print("Policy pipeline demo (reflector + summarization rules)")
    print("=" * 60)

    # -- Custom reflector callable for the policy engine ---------------
    # The policy engine's _derive_summary calls:
    #   reflector(group: list[MemoryRecord], rule: SummarizationRule, now: datetime)
    # and expects a dict with at least a "content" key.
    def my_reflector(
        records: list[MemoryRecord],
        rule: Any,
        now: datetime,
    ) -> dict[str, Any]:
        themes = set()
        for rec in records:
            for word in rec.content.lower().split():
                cleaned = word.strip(".,!?;:'\"()[]")
                if len(cleaned) > 4:
                    themes.add(cleaned)
        top = sorted(themes)[:6]
        return {
            "content": f"Learned pattern from {len(records)} episodes: {', '.join(top)}.",
            "summary_model": "custom-mock",
            "distillation_kind": "semantic",
        }

    # -- Build config with extraction + summarization rules -----------
    config = EngramConfig.from_mapping({
        "driver": {"kind": "memory"},
        "policies": {
            "extraction": [
                {
                    "name": "capture_deployments",
                    "trigger": "deployment_event",
                    "create": {
                        "type": "episodic",
                        "scope": "user",
                        "payload": {"action_signature": "deployment"},
                    },
                },
            ],
            "summarization": [
                {
                    "name": "distill_deployments",
                    "applies_to": {"type": "episodic"},
                    "trigger": "same_action_count >= 2",
                    "promote_to": {"type": "semantic", "scope": "agent"},
                },
            ],
        },
    })

    # Attach our custom reflector to the summarization rule
    for rule in config.policies.summarization:
        rule.reflector = my_reflector

    # -- Create the client with this config ----------------------------
    client = MemoryClient(config=config)

    # -- Ingest a series of deployment events --------------------------
    events = [
        {
            "type": "deployment_event",
            "content": "Deployed authentication service v1.0 to staging.",
            "source": "ci-pipeline",
            "user_id": "bob",
            "metadata": {"payload": {"action_signature": "deployment"}},
        },
        {
            "type": "deployment_event",
            "content": "Deployed authentication service v1.1 to production.",
            "source": "ci-pipeline",
            "user_id": "bob",
            "metadata": {"payload": {"action_signature": "deployment"}},
        },
        {
            "type": "deployment_event",
            "content": "Deployed payment gateway v3.0 to staging.",
            "source": "ci-pipeline",
            "user_id": "bob",
            "metadata": {"payload": {"action_signature": "deployment"}},
        },
    ]

    print("\nIngesting deployment events...")
    for i, event in enumerate(events, 1):
        outcome = await client.ingest_event(event)
        print(f"  Event {i}: extracted={len(outcome.extracted)}, "
              f"promoted={len(outcome.promoted)}, "
              f"policies_fired={outcome.policies_fired}")

    # -- Inspect what got stored ---------------------------------------
    all_records = await client.list()
    print(f"\nAll stored records ({len(all_records)}):")
    for rec in all_records:
        print(f"  [{rec.type:11s}] {rec.content[:70]}")
        if rec.payload:
            kind = rec.payload.get("distillation_kind")
            model = rec.payload.get("summary_model")
            sources = rec.payload.get("source_record_ids", [])
            if kind:
                print(f"                 ^ distillation_kind={kind}, "
                      f"summary_model={model}, sources={len(sources)}")

    # -- You can also run promote() to re-scan and create new summaries
    newly_promoted = await client.promote()
    print(f"\nPromote pass: {len(newly_promoted)} new record(s) created")
    for rec in newly_promoted:
        print(f"  [{rec.type:11s}] {rec.content[:70]}")

    # -- Final state ---------------------------------------------------
    final = await client.list()
    episodic_count = sum(1 for r in final if r.type == "episodic")
    semantic_count = sum(1 for r in final if r.type == "semantic")
    procedural_count = sum(1 for r in final if r.type == "procedural")
    print(f"\nFinal memory store: "
          f"{episodic_count} episodic, "
          f"{semantic_count} semantic, "
          f"{procedural_count} procedural")
    print()


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    standalone_reflector_demo()
    asyncio.run(policy_pipeline_demo())
