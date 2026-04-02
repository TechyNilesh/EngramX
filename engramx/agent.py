"""Agent runner with memory attribution."""

from __future__ import annotations

import importlib
from typing import Any

from .client import MemoryClient
from .observability import OutputAttribution, ScoredMemory


def _dependency_available(module_name: str) -> bool:
    """Check whether an optional dependency is importable."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def _build_context_prompt(
    task: str,
    memories: list[ScoredMemory],
) -> str:
    """Construct the system + user prompt incorporating memory context."""
    if memories:
        memory_lines = "\n".join(
            f"- (score {m.score:.3f}) {m.record.content}" for m in memories
        )
    else:
        memory_lines = "- No relevant memories found."

    return (
        "You are an AI assistant with access to the user's memory store. "
        "Use the retrieved memories below as context to answer the task. "
        "If a memory is not relevant, ignore it.\n\n"
        f"### Retrieved Memories\n{memory_lines}\n\n"
        f"### Task\n{task}"
    )


class EngramAgent:
    """Lightweight agent that enriches LLM calls with Engram memory context.

    Parameters
    ----------
    client:
        An initialised :class:`MemoryClient` used to search memories.
    llm_client:
        An optional pre-configured LLM client (e.g. ``openai.AsyncOpenAI``).
        When *None*, a default ``openai.AsyncOpenAI`` client is created on
        first use.
    model:
        The chat model to call (default ``"gpt-4o-mini"``).
    """

    def __init__(
        self,
        client: MemoryClient,
        llm_client: Any = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.client = client
        self.model = model
        self._llm_client = llm_client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm_client(self) -> Any:
        """Lazily resolve the async OpenAI client."""
        if self._llm_client is not None:
            return self._llm_client

        if not _dependency_available("openai"):
            raise ImportError(
                "The 'openai' package is required for EngramAgent's default LLM. "
                "Install it with: pip install openai"
            )
        import openai

        self._llm_client = openai.AsyncOpenAI()
        return self._llm_client

    async def _call_llm(self, prompt: str) -> str:
        """Send *prompt* to the configured LLM and return the response text."""
        llm = self._get_llm_client()
        response = await llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_with_attribution(
        self,
        task: str,
        user_id: str | None = None,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        access_roles: Any | None = None,
    ) -> tuple[str, OutputAttribution]:
        """Run *task* against the LLM enriched with relevant memories.

        Returns
        -------
        tuple[str, OutputAttribution]
            The LLM response text and an :class:`OutputAttribution` recording
            which memories were used and which policies fired.
        """
        # 1. Search relevant memories
        search_filters = dict(filters or {})
        if user_id is not None:
            search_filters.setdefault("user_id", user_id)

        memories: list[ScoredMemory] = await self.client.search(
            task,
            filters=search_filters,
            top_k=top_k,
            access_roles=access_roles,
        )

        # 2. Build prompt with memory context
        prompt = _build_context_prompt(task, memories)

        # 3. Call LLM
        response_text = await self._call_llm(prompt)

        # 4. Build attribution
        attribution = self.client.build_attribution(memories=memories)

        return response_text, attribution
