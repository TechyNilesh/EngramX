from __future__ import annotations

import importlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence

from .config import GovernanceRule, RetentionRule, SummarizationRule
from .models import MemorySignal
from .schema import MemoryRecord, new_memory_id, parse_datetime, utcnow

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
TRIGGER_COUNT_RE = re.compile(r"^same_[a-z0-9_]+_count\s*>=\s*(\d+)\s*$")


@dataclass(slots=True)
class GovernanceAuditEntry:
    memory_id: str
    rule_name: str
    action: str
    reason: str
    timestamp: datetime
    details: dict[str, Any] = field(default_factory=dict)


def _to_datetime(value: datetime | str | None) -> datetime:
    return parse_datetime(value) or utcnow()


def _matches_selector(record: MemoryRecord, selector: dict[str, Any]) -> bool:
    for key, expected in selector.items():
        actual = getattr(record, key, None)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _content_matches(content: str, patterns: Sequence[str]) -> bool:
    lowered = content.lower()
    return all(pattern.lower() in lowered for pattern in patterns)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(text)}


def _normalize_content(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def _default_group_signature(record: MemoryRecord) -> str:
    payload = record.payload or {}
    for key in ("action_signature", "action", "summary_signature", "cluster_signature", "topic"):
        value = payload.get(key)
        if value:
            return str(value)
    return _normalize_content(record.content)


def _rule_threshold(trigger: str) -> int | None:
    match = TRIGGER_COUNT_RE.match(trigger.strip())
    if not match:
        return None
    return int(match.group(1))


def _resolve_reflector(reflector: Any) -> Callable[[list[MemoryRecord], SummarizationRule, datetime], Any] | None:
    if reflector is None:
        return None
    if callable(reflector):
        return reflector
    if isinstance(reflector, str) and reflector:
        module_name, _, attr = reflector.rpartition(":")
        if not module_name:
            module_name, _, attr = reflector.rpartition(".")
        if not module_name or not attr:
            raise ValueError(f"Unsupported reflector path: {reflector!r}")
        module = importlib.import_module(module_name)
        resolved = getattr(module, attr)
        if not callable(resolved):
            raise TypeError(f"Reflector {reflector!r} is not callable")
        return resolved
    raise TypeError(f"Unsupported reflector value: {reflector!r}")


def _semantic_summary_from_group(
    group: list[MemoryRecord],
    rule: SummarizationRule,
    *,
    now: datetime,
) -> tuple[str, dict[str, Any]]:
    token_counts: dict[str, int] = defaultdict(int)
    for record in group:
        for token in _tokenize(record.content):
            token_counts[token] += 1
    common_tokens = [token for token, count in sorted(token_counts.items()) if count == len(group)]
    highlighted = common_tokens[:6] or sorted(token_counts, key=token_counts.get, reverse=True)[:6]
    content = rule.promote_to.get("content")
    if content:
        summary_text = str(content)
    elif highlighted:
        summary_text = f"Distilled semantic rule from {len(group)} episodes: {' '.join(highlighted)}."
    else:
        summary_text = f"Distilled semantic rule from {len(group)} episodes."
    payload = {
        "source_record_ids": [item.id for item in group],
        "distilled_from": [item.id for item in group],
        "distillation_kind": "semantic",
        "summary_model": rule.summary_model,
        "rule_name": rule.name,
        "generated_at": now.isoformat(),
        "keywords": highlighted,
    }
    return summary_text, payload


def _procedural_summary_from_group(
    group: list[MemoryRecord],
    rule: SummarizationRule,
    *,
    now: datetime,
) -> tuple[str, dict[str, Any]]:
    content = rule.promote_to.get("content")
    if content:
        summary_text = str(content)
    else:
        summary_text = f"Learned procedural pattern from {len(group)} successful episodes."
    payload = {
        "source_record_ids": [item.id for item in group],
        "distilled_from": [item.id for item in group],
        "distillation_kind": "procedural",
        "summary_model": rule.summary_model,
        "rule_name": rule.name,
        "generated_at": now.isoformat(),
    }
    return summary_text, payload


def _derive_summary(
    group: list[MemoryRecord],
    rule: SummarizationRule,
    *,
    now: datetime,
) -> tuple[str, dict[str, Any]]:
    reflector = _resolve_reflector(rule.reflector)
    if reflector is not None:
        result = reflector(group, rule, now)
        if isinstance(result, MemoryRecord):
            payload = result.to_dict()
            payload.setdefault("source_record_ids", [item.id for item in group])
            payload.setdefault("distilled_from", [item.id for item in group])
            payload.setdefault("rule_name", rule.name)
            return result.content, payload
        if isinstance(result, dict):
            content = str(result.get("content") or result.get("summary") or "")
            payload = dict(result)
            payload.setdefault("source_record_ids", [item.id for item in group])
            payload.setdefault("distilled_from", [item.id for item in group])
            payload.setdefault("rule_name", rule.name)
            return content or _procedural_summary_from_group(group, rule, now=now)[0], payload
        if result is not None:
            content = str(result)
            return content, {
                "source_record_ids": [item.id for item in group],
                "distilled_from": [item.id for item in group],
                "rule_name": rule.name,
                "summary_model": rule.summary_model,
            }

    target_type = rule.promote_to.get("type", "procedural")
    if target_type == "semantic":
        return _semantic_summary_from_group(group, rule, now=now)
    return _procedural_summary_from_group(group, rule, now=now)


def _safe_formula(formula: str, signal: MemorySignal) -> float:
    allowed_names = {
        "retry_count": signal.retry_count,
        "content_length": len(signal.content),
    }
    try:
        value = eval(formula, {"__builtins__": {}}, allowed_names)
    except Exception:
        return 0.5
    return max(0.0, min(1.0, float(value)))


def build_memory_record(
    *,
    type: str,
    scope: str,
    content: str,
    source: str,
    signal: MemorySignal,
    create: dict[str, Any],
    derived_from: list[str] | None = None,
    provenance: list[str] | None = None,
    timestamp: datetime | None = None,
) -> MemoryRecord:
    now = _to_datetime(timestamp or signal.timestamp)
    importance = create.get("importance_score")
    if importance is None and "importance_score_formula" in create:
        importance = _safe_formula(str(create["importance_score_formula"]), signal)
    payload = dict(create.get("payload") or {})
    payload.update(signal.metadata.get("payload", {}))
    return MemoryRecord(
        id=new_memory_id(),
        type=type,
        scope=scope,
        content=create.get("content", content),
        agent_id=create.get("agent_id", signal.agent_id),
        user_id=create.get("user_id", signal.user_id),
        session_id=create.get("session_id", signal.session_id),
        payload=payload or None,
        timestamp_created=now,
        timestamp_updated=now,
        valid_from=_to_datetime(create["valid_from"]) if create.get("valid_from") else None,
        valid_to=_to_datetime(create["valid_to"]) if create.get("valid_to") else None,
        source=create.get("source", source),
        provenance=provenance or [str(signal.metadata.get("event_id", signal.type))],
        derived_from=derived_from or [],
        importance_score=float(importance if importance is not None else 0.5),
        sensitivity_level=create.get("sensitivity_level", "public"),
        retention_policy=create.get("retention_policy"),
        access_roles=list(create.get("access_roles", [])),
        embedding_model=create.get("embedding_model"),
    )


def extract_records(
    rules: Iterable[Any],
    signal: MemorySignal,
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    extracted: list[MemoryRecord] = []
    timestamp = _to_datetime(now or signal.timestamp)
    for rule in rules:
        if rule.trigger != signal.type:
            continue
        conditions = rule.conditions or []
        matches = True
        for condition in conditions:
            if "content_matches" in condition and not _content_matches(signal.content, condition["content_matches"]):
                matches = False
                break
        if not matches:
            continue
        create = dict(rule.create)
        extracted.append(
            build_memory_record(
                type=create.get("type", "episodic"),
                scope=create.get("scope", "session"),
                content=signal.content,
                source=signal.source,
                signal=signal,
                create=create,
                timestamp=timestamp,
            )
        )
    return extracted


def _group_records_for_rule(
    records: Iterable[MemoryRecord],
    rule: SummarizationRule,
) -> dict[str, list[MemoryRecord]]:
    grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        if not _matches_selector(record, rule.applies_to):
            continue
        if record.type != "episodic":
            continue
        payload = record.payload or {}
        if payload.get("success") is False and "success" in rule.trigger:
            continue
        grouped[_default_group_signature(record)].append(record)
    return grouped


def summarize_records(
    records: Iterable[MemoryRecord],
    rules: Iterable[SummarizationRule],
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    current = _to_datetime(now)
    promoted: list[MemoryRecord] = []
    for rule in rules:
        threshold = _rule_threshold(rule.trigger)
        if threshold is None:
            continue
        grouped = _group_records_for_rule(records, rule)
        for _, group in grouped.items():
            if len(group) < threshold:
                continue
            content, payload = _derive_summary(group, rule, now=current)
            target_type = rule.promote_to.get("type", "procedural")
            promoted.append(
                MemoryRecord(
                    id=new_memory_id(),
                    type=target_type,
                    scope=rule.promote_to.get("scope", "agent"),
                    content=content,
                    agent_id=group[0].agent_id,
                    user_id=group[0].user_id,
                    session_id=group[0].session_id,
                    payload=payload,
                    timestamp_created=current,
                    timestamp_updated=current,
                    source="reflection",
                    provenance=[item.id for item in group],
                    derived_from=[item.id for item in group],
                    importance_score=max(item.importance_score for item in group),
                    sensitivity_level=group[0].sensitivity_level,
                    retention_policy=group[0].retention_policy,
                    access_roles=list(group[0].access_roles),
                    valid_from=group[0].valid_from,
                    valid_to=group[0].valid_to,
                )
            )
    return promoted


def promote_repeated_success(
    records: Iterable[MemoryRecord],
    rules: Iterable[SummarizationRule],
    *,
    now: datetime | None = None,
) -> list[MemoryRecord]:
    return summarize_records(records, rules, now=now)


def compute_decay_factor(
    record: MemoryRecord,
    rules: Iterable[RetentionRule],
    *,
    now: datetime | None = None,
) -> float:
    current = _to_datetime(now)
    applicable = [rule for rule in rules if _matches_selector(record, rule.applies_to)]
    if not applicable:
        return record.decay_factor
    age_days = max((current - _to_datetime(record.timestamp_created)).total_seconds() / 86400.0, 0.0)
    factors: list[float] = []
    for rule in applicable:
        if rule.decay_function == "none" or rule.half_life_days is None:
            factors.append(1.0)
            continue
        half_life = max(float(rule.half_life_days), 0.001)
        if rule.decay_function == "exponential":
            factor = 0.5 ** (age_days / half_life)
        elif rule.decay_function == "linear":
            factor = max(0.0, 1.0 - (age_days / half_life))
        elif rule.decay_function == "step":
            factor = 0.5 if age_days >= half_life else 1.0
        else:
            factor = 1.0
        factors.append(max(rule.floor_score, factor))
    return min(factors) if factors else record.decay_factor


def _apply_governance_deletion(
    records: Iterable[MemoryRecord],
    rules: Iterable[GovernanceRule],
    *,
    user_id: str | None = None,
    reason: str = "user_deletion",
    now: datetime | None = None,
) -> tuple[list[MemoryRecord], int, list[GovernanceAuditEntry]]:
    current = _to_datetime(now)
    kept: list[MemoryRecord] = []
    deleted = 0
    audit_log: list[GovernanceAuditEntry] = []
    for record in records:
        should_delete = False
        matched_rule: GovernanceRule | None = None
        matched_any = False
        for rule in rules:
            if not _matches_selector(record, rule.applies_to):
                continue
            matched_any = True
            if user_id is not None and record.user_id != user_id:
                continue
            matched_rule = rule
            if reason == "user_deletion" and rule.on_user_deletion == "delete_all":
                should_delete = True
            elif rule.retention_days is not None:
                age_days = (current - _to_datetime(record.timestamp_created)).total_seconds() / 86400.0
                if age_days >= rule.retention_days:
                    should_delete = True
            audit_log.append(
                GovernanceAuditEntry(
                    memory_id=record.id,
                    rule_name=rule.name,
                    action="delete" if should_delete else "retain",
                    reason=reason,
                    timestamp=current,
                    details={
                        "retention_days": rule.retention_days,
                        "on_user_deletion": rule.on_user_deletion,
                        "audit_log": rule.audit_log,
                        "sensitivity_level": record.sensitivity_level,
                    },
                )
            )
            if should_delete:
                deleted += 1
                break
        if not should_delete:
            kept.append(record)
        elif matched_rule is None and matched_any:
            audit_log.append(
                GovernanceAuditEntry(
                    memory_id=record.id,
                    rule_name="unmatched",
                    action="delete",
                    reason=reason,
                    timestamp=current,
                    details={"reason": "matched_selector_but_no_retention_action"},
                )
            )
    return kept, deleted, audit_log


def apply_governance_deletion(
    records: Iterable[MemoryRecord],
    rules: Iterable[GovernanceRule],
    *,
    user_id: str | None = None,
    reason: str = "user_deletion",
    now: datetime | None = None,
    include_audit_log: bool = False,
) -> tuple[list[MemoryRecord], int] | tuple[list[MemoryRecord], int, list[GovernanceAuditEntry]]:
    kept, deleted, audit_log = _apply_governance_deletion(
        records,
        rules,
        user_id=user_id,
        reason=reason,
        now=now,
    )
    if include_audit_log:
        return kept, deleted, audit_log
    return kept, deleted
