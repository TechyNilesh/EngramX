from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(slots=True)
class DriverConfig:
    kind: str = "memory"
    path: str = ":memory:"


@dataclass(slots=True)
class ExtractionRule:
    name: str
    trigger: str
    create: dict[str, Any] = field(default_factory=dict)
    conditions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class RetentionRule:
    name: str
    applies_to: dict[str, Any] = field(default_factory=dict)
    decay_function: str = "none"
    half_life_days: float | None = None
    floor_score: float = 0.0


@dataclass(slots=True)
class SummarizationRule:
    name: str
    applies_to: dict[str, Any] = field(default_factory=dict)
    trigger: str = ""
    promote_to: dict[str, Any] = field(default_factory=dict)
    summary_model: str | None = None
    reflector: Any | None = None


@dataclass(slots=True)
class GovernanceRule:
    name: str
    applies_to: dict[str, Any] = field(default_factory=dict)
    retention_days: int | None = None
    on_user_deletion: str = "retain"
    audit_log: bool = False


@dataclass(slots=True)
class PolicyConfig:
    extraction: list[ExtractionRule] = field(default_factory=list)
    retention: list[RetentionRule] = field(default_factory=list)
    summarization: list[SummarizationRule] = field(default_factory=list)
    governance: list[GovernanceRule] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PolicyConfig":
        payload = dict(data or {})
        if "policies" in payload:
            payload = dict(payload.get("policies") or {})
        return cls(
            extraction=[ExtractionRule(**item) for item in payload.get("extraction", [])],
            retention=[RetentionRule(**item) for item in payload.get("retention", [])],
            summarization=[SummarizationRule(**item) for item in payload.get("summarization", [])],
            governance=[GovernanceRule(**item) for item in payload.get("governance", [])],
        )

    def has_rules(self) -> bool:
        return any((self.extraction, self.retention, self.summarization, self.governance))


@dataclass(slots=True)
class EngramConfig:
    driver: DriverConfig = field(default_factory=DriverConfig)
    policies: PolicyConfig = field(default_factory=PolicyConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "EngramConfig":
        payload = dict(data or {})
        driver_data = dict(payload.get("driver") or {})
        policies = PolicyConfig.from_mapping(payload.get("policies") or payload)
        return cls(driver=DriverConfig(**driver_data), policies=policies)


def _load_mapping(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    path = Path(source)
    if path.exists():
        text = path.read_text()
    else:
        text = str(source)
    return yaml.safe_load(text) or {}


def load_policy_config(source: str | Path | Mapping[str, Any]) -> PolicyConfig:
    return PolicyConfig.from_mapping(_load_mapping(source))


def load_config(source: str | Path | Mapping[str, Any]) -> EngramConfig:
    return EngramConfig.from_mapping(_load_mapping(source))
