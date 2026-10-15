"""双场景目录和任务集的稳定数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class EvidenceRecord:
    source_seed: int
    derived_rule: str
    synthetic: bool
    unit: str
    version: str
    review_status: str = "synthetic-unreviewed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TypedPort:
    port_id: str
    port_type: str
    direction: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComponentTemplate:
    template_id: str
    scene: str
    role: str
    parameters: dict[str, float]
    parameter_units: dict[str, str]
    ports: tuple[TypedPort, ...]
    constraints: tuple[str, ...]
    evidence: EvidenceRecord

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "scene": self.scene,
            "role": self.role,
            "parameters": dict(sorted(self.parameters.items())),
            "parameter_units": dict(sorted(self.parameter_units.items())),
            "ports": [_plain(port) for port in self.ports],
            "constraints": list(self.constraints),
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class ConfigurationTask:
    task_id: str
    scene: str
    split: str
    seed: int
    task_type: str
    requirement: str
    required_roles: tuple[str, ...]
    candidate_ids: dict[str, tuple[str, ...]]
    numeric_requirements: dict[str, float]
    injected_conflict: str | None = None
    conflict_scope: str | None = None
    expected_feasible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scene": self.scene,
            "split": self.split,
            "seed": self.seed,
            "task_type": self.task_type,
            "requirement": self.requirement,
            "required_roles": list(self.required_roles),
            "candidate_ids": {
                role: list(ids) for role, ids in sorted(self.candidate_ids.items())
            },
            "numeric_requirements": dict(sorted(self.numeric_requirements.items())),
            "injected_conflict": self.injected_conflict,
            "conflict_scope": self.conflict_scope,
            "expected_feasible": self.expected_feasible,
        }


@dataclass(frozen=True)
class TruthRecord:
    task_id: str
    feasible: bool
    selected_candidate_ids: dict[str, str] | None
    violated_constraints: tuple[str, ...]
    checked_combinations: int
    full_catalog_solution_count: int
    eligible_for_feasibility_rate: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "feasible": self.feasible,
            "selected_candidate_ids": self.selected_candidate_ids,
            "violated_constraints": list(self.violated_constraints),
            "checked_combinations": self.checked_combinations,
            "full_catalog_solution_count": self.full_catalog_solution_count,
            "eligible_for_feasibility_rate": self.eligible_for_feasibility_rate,
        }


@dataclass(frozen=True)
class FreezeManifest:
    seed: int
    catalog_count: int
    task_counts: dict[str, int]
    counts: dict[str, int]
    scale_status: str
    formal_target: dict[str, int]
    files: dict[str, str]
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "catalog_count": self.catalog_count,
            "task_counts": dict(sorted(self.task_counts.items())),
            "counts": dict(sorted(self.counts.items())),
            "scale_status": self.scale_status,
            "formal_target": dict(sorted(self.formal_target.items())),
            "files": dict(sorted(self.files.items())),
            "manifest_sha256": self.manifest_sha256,
        }
