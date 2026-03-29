"""LLM-powered reflection for summarization policies."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from typing import Any

from .schema import MemoryRecord


def _dependency_available(module_name: str) -> bool:
    """Check whether an optional dependency is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _build_prompt(records: list[MemoryRecord], rule: Any, now: datetime) -> str:
    """Build a summarization prompt from episodic memory records."""
    record_texts = []
    for i, rec in enumerate(records, 1):
        created = rec.timestamp_created.isoformat() if rec.timestamp_created else "unknown"
        record_texts.append(f"{i}. [{created}] {rec.content}")

    records_block = "\n".join(record_texts) or "(no records)"

    rule_description = ""
    if rule is not None:
        rule_description = f"\nSummarization rule: {rule}\n"

    return (
        "You are a memory reflection engine. Your job is to distill a set of "
        "episodic memory records into a single, concise summary that preserves "
        "the most important information.\n"
        f"\nCurrent time: {now.isoformat()}"
        f"{rule_description}"
        f"\n--- Episodic Records ---\n{records_block}\n"
        "--- End Records ---\n\n"
        "Produce a JSON object with exactly one key \"content\" whose value is "
        "the distilled summary string. Respond ONLY with valid JSON."
    )


def _parse_llm_response(raw: str) -> str:
    """Extract the summary content from the LLM's JSON response."""
    raw = raw.strip()
    # Try JSON parsing first
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "content" in data:
            return str(data["content"])
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: use the raw text as the summary
    return raw


class LLMReflector:
    """Base reflector that calls an LLM to summarise episodic records.

    Subclasses override :meth:`_call_llm` for specific providers.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        records: list[MemoryRecord],
        rule: Any = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Reflect on *records* and return a summary dict."""
        now = now or datetime.now(timezone.utc)
        prompt = _build_prompt(records, rule, now)
        raw_response = self._call_llm(prompt)
        content = _parse_llm_response(raw_response)
        return {
            "content": content,
            "summary_model": self.model,
            "provider": self.provider,
            "source_count": len(records),
            "source_ids": [r.id for r in records],
        }

    # ------------------------------------------------------------------
    # Override point
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError(
            "LLMReflector is abstract; use OpenAIReflector or AnthropicReflector, "
            "or call create_reflector()."
        )


class OpenAIReflector(LLMReflector):
    """Reflector that uses the OpenAI chat completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        if not _dependency_available("openai"):
            raise ImportError(
                "The 'openai' package is required for OpenAIReflector. "
                "Install it with: pip install openai"
            )
        super().__init__(provider="openai", model=model, api_key=api_key)

        import openai

        self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

    def _call_llm(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""


class AnthropicReflector(LLMReflector):
    """Reflector that uses the Anthropic messages API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        if not _dependency_available("anthropic"):
            raise ImportError(
                "The 'anthropic' package is required for AnthropicReflector. "
                "Install it with: pip install anthropic"
            )
        super().__init__(provider="anthropic", model=model, api_key=api_key)

        import anthropic

        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )

    def _call_llm(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        # Anthropic returns a list of content blocks
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        )


def create_reflector(provider: str = "openai", **kwargs: Any) -> LLMReflector:
    """Factory that returns the appropriate :class:`LLMReflector` implementation.

    Parameters
    ----------
    provider:
        One of ``"openai"`` (default) or ``"anthropic"``.
    **kwargs:
        Forwarded to the chosen reflector constructor (``model``, ``api_key``, etc.).
    """
    if provider == "openai":
        return OpenAIReflector(**kwargs)
    if provider == "anthropic":
        return AnthropicReflector(**kwargs)
    raise ValueError(f"Unknown reflector provider: {provider!r}")
