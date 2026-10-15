from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_EVIDENCE_LEVELS = {
    "source_supported_requirement",
    "design_assumption",
    "derived_from_documented_assumptions",
    "source_conflict",
    "unverified_claim",
}
_EVALUATION_USES = {
    "scenario_constraint",
    "interview_checklist",
    "sensitivity_parameter",
    "context_only",
    "exclude_from_evaluation",
}
_EXCLUDED_EVIDENCE_LEVELS = {"source_conflict", "unverified_claim"}


@dataclass(frozen=True)
class CaseAuditReport:
    case_id: str
    source_class: str
    formal_evaluation_eligible: bool
    blocking_requirements: tuple[str, ...]
    valid: bool
    violating_field_ids: tuple[str, ...]
    errors: tuple[str, ...]


def _load_mapping(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name} must contain a YAML mapping")
    return loaded


def audit_external_case(case_dir: Path) -> CaseAuditReport:
    """Check evidence labels without reading or changing benchmark artefacts."""
    metadata = _load_mapping(case_dir / "case_metadata.yaml")
    field_register = _load_mapping(case_dir / "field_register.yaml")

    errors: list[str] = []
    violations: list[str] = []
    case_id = str(metadata.get("case_id", ""))
    source_class = str(metadata.get("source_class", ""))
    eligible = metadata.get("formal_evaluation_eligible")
    blockers = metadata.get("blocking_requirements", [])

    if not case_id:
        errors.append("case_id is required")
    if not source_class:
        errors.append("source_class is required")
    if not isinstance(eligible, bool):
        errors.append("formal_evaluation_eligible must be boolean")
        eligible = False
    if not isinstance(blockers, list) or not all(
        isinstance(item, str) and item for item in blockers
    ):
        errors.append("blocking_requirements must be a list of non-empty strings")
        blockers = []

    fields = field_register.get("fields")
    if not isinstance(fields, list) or not fields:
        errors.append("field_register must contain a non-empty fields list")
        fields = []

    for field in fields:
        if not isinstance(field, dict):
            errors.append("each field entry must be a mapping")
            continue
        field_id = str(field.get("field_id", ""))
        evidence_level = field.get("evidence_level")
        evaluation_use = field.get("evaluation_use")
        if not field_id:
            errors.append("field_id is required")
            continue
        invalid = False
        if evidence_level not in _EVIDENCE_LEVELS:
            errors.append(f"{field_id}: invalid evidence_level")
            invalid = True
        if evaluation_use not in _EVALUATION_USES:
            errors.append(f"{field_id}: invalid evaluation_use")
            invalid = True
        if evidence_level in _EXCLUDED_EVIDENCE_LEVELS and (
            evaluation_use != "exclude_from_evaluation"
        ):
            errors.append(f"{field_id}: unsupported evidence must be excluded")
            invalid = True
        if evidence_level in {
            "design_assumption",
            "derived_from_documented_assumptions",
        } and evaluation_use == "formal_evaluation_input":
            errors.append(f"{field_id}: assumption-derived values cannot be formal input")
            invalid = True
        if not eligible and evaluation_use == "formal_evaluation_input":
            errors.append(f"{field_id}: ineligible case cannot provide formal input")
            invalid = True
        if invalid:
            violations.append(field_id)

    if source_class == "document_derived_engineering_case" and eligible:
        errors.append("document-derived cases must remain ineligible until upgraded")

    return CaseAuditReport(
        case_id=case_id,
        source_class=source_class,
        formal_evaluation_eligible=bool(eligible),
        blocking_requirements=tuple(blockers),
        valid=not errors,
        violating_field_ids=tuple(sorted(set(violations))),
        errors=tuple(errors),
    )
