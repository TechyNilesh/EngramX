from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .schema import MemoryRecord


@dataclass(slots=True)
class ScoredMemory:
    record: MemoryRecord
    score: float
    lexical_score: float
    importance_component: float
    decay_component: float
    matched_terms: list[str] = field(default_factory=list)
    policy_filters: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemoryTimelineEvent:
    operation: str
    memory_id: str
    timestamp: datetime
    actor: str = "system"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedMemory:
    id: str
    score: float
    decay_adjusted: float
    matched_terms: list[str]
    policy_filters: list[str]


@dataclass(slots=True)
class MemoryTrace:
    query: str
    filters: dict[str, Any]
    retrieved: list[RetrievedMemory] = field(default_factory=list)
    candidates: list[ScoredMemory] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Memory Trace",
            f"- Query: `{self.query}`",
            f"- Filters: `{self.filters}`",
            "",
        ]
        if not self.retrieved:
            lines.append("No memories retrieved.")
            return "\n".join(lines)
        lines.extend(
            [
                "| Memory ID | Score | Decay Adjusted | Matched Terms | Policy Filters |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for item in self.retrieved:
            lines.append(
                f"| `{item.id}` | {item.score:.3f} | {item.decay_adjusted:.3f} | "
                f"{', '.join(item.matched_terms) or '-'} | {', '.join(item.policy_filters) or '-'} |"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class OutputAttribution:
    memories_used: list[str] = field(default_factory=list)
    policies_fired: list[str] = field(default_factory=list)
