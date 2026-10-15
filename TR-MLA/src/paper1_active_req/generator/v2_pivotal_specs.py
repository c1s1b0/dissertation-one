from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.solver.engine import ConstraintEngine

from .complete_specs import CompleteSpec, DatasetManifest, RequirementTask, generate_complete_spec


Mechanism = Literal[
    "interface_unlock", "interval_reduction", "conditional_correlation"
]


class V2AuditMetadata(BaseModel):
    module_id: str
    pivot_field_id: str | None
    affected_critical_field_ids: tuple[str, ...] = ()
    mechanism: Mechanism


class V2RequirementTask(RequirementTask):
    audit: V2AuditMetadata

    def ground_truth_to_requirement_state(self) -> RequirementState:
        return RequirementState(
            values={
                field_id: value.model_copy(deep=True)
                for field_id, value in self.ground_truth.items()
            }
        )


_MECHANISMS: tuple[tuple[Mechanism, str, str, tuple[str, ...]], ...] = (
    (
        "interface_unlock",
        "upstream_downstream_interface",
        "upstream_output_interface",
        ("station_input_interface",),
    ),
    (
        "interval_reduction",
        "load_space",
        "side_clearance_mm",
        ("available_width_mm",),
    ),
    (
        "conditional_correlation",
        "positioning_speed",
        "workpiece_fragile",
        ("positioning_accuracy_mm", "repeatability_mm"),
    ),
)


def _write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(item.model_dump_json() for item in items) + "\n",
        encoding="utf-8",
    )


def _mask_v2_spec(
    spec: CompleteSpec,
    missing_rate: float,
    rng: np.random.Generator,
    audit: V2AuditMetadata,
) -> V2RequirementTask:
    candidates = sorted(
        field_id for field_id, value in spec.values.items() if field_id != "power_supply"
    )
    hidden_count = round(len(candidates) * missing_rate)
    hidden_ids = set(rng.choice(candidates, size=hidden_count, replace=False).tolist())
    required_hidden = {audit.pivot_field_id, *audit.affected_critical_field_ids}
    required_hidden.discard(None)
    for field_id in sorted(required_hidden):
        if field_id in hidden_ids:
            continue
        removable = sorted(hidden_ids - required_hidden)
        if not removable:
            raise RuntimeError("missing-rate budget cannot accommodate pivotal fields")
        hidden_ids.remove(removable[0])
        hidden_ids.add(field_id)

    initial_values = {
        field_id: value.model_copy(deep=True)
        for field_id, value in spec.values.items()
    }
    for field_id in hidden_ids:
        initial_values[field_id] = FieldValue(
            value=None, status=FieldStatus.HIDDEN, source="v2_mask"
        )
    return V2RequirementTask(
        task_id=f"{spec.spec_id}__m{int(missing_rate * 100)}",
        spec_id=spec.spec_id,
        split=spec.split,
        missing_rate=missing_rate,
        initial_state=RequirementState(values=initial_values),
        ground_truth={
            field_id: value.model_copy(deep=True)
            for field_id, value in spec.values.items()
        },
        audit=audit,
    )


def _audit_for_spec(index: int) -> V2AuditMetadata:
    mechanism, module_id, pivot, affected = _MECHANISMS[index % len(_MECHANISMS)]
    if index % 4 == 0:
        return V2AuditMetadata(
            module_id=module_id,
            pivot_field_id=None,
            affected_critical_field_ids=(),
            mechanism=mechanism,
        )
    return V2AuditMetadata(
        module_id=module_id,
        pivot_field_id=pivot,
        affected_critical_field_ids=affected,
        mechanism=mechanism,
    )


def _apply_pivotal_relation(
    spec: CompleteSpec, audit: V2AuditMetadata
) -> CompleteSpec:
    values = {
        field_id: value.model_copy(deep=True)
        for field_id, value in spec.values.items()
    }
    if audit.mechanism == "interval_reduction" and audit.pivot_field_id:
        values["available_width_mm"] = FieldValue(
            value=(
                float(values["workpiece_width_mm"].value)
                + 2.0 * float(values["side_clearance_mm"].value)
                + 1.0
            ),
            status=FieldStatus.OBSERVED,
            source="v2_pivotal_generation",
        )
    if audit.mechanism == "conditional_correlation" and audit.pivot_field_id:
        fragile = bool(values["workpiece_fragile"].value)
        values["positioning_accuracy_mm"] = FieldValue(
            value=0.2 if fragile else 2.0,
            status=FieldStatus.OBSERVED,
            source="v2_pivotal_generation",
        )
        values["repeatability_mm"] = FieldValue(
            value=0.1 if fragile else 1.0,
            status=FieldStatus.OBSERVED,
            source="v2_pivotal_generation",
        )
    return spec.model_copy(update={"values": values})


def generate_v2_dataset(
    seed: int,
    output_dir: Path,
    schema_path: Path,
) -> DatasetManifest:
    """Generate the independent, fixed-size v2 dataset without v1 artifacts."""
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(seed)
    split_counts = {"train": 108, "validation": 36, "test": 36}
    specs: list[CompleteSpec] = []
    split_specs: dict[str, list[CompleteSpec]] = {name: [] for name in split_counts}
    index = 0
    for split, count in split_counts.items():
        for _ in range(count):
            audit = _audit_for_spec(index)
            spec = generate_complete_spec(
                rng, schema, engine, f"v2-spec-{index:04d}"
            )
            spec = _apply_pivotal_relation(spec, audit)
            if engine.check(spec.to_requirement_state()).satisfiable is not True:
                raise RuntimeError(f"v2 pivotal relation is unsatisfiable: {spec.spec_id}")
            spec.split = split
            specs.append(spec)
            split_specs[split].append(spec)
            index += 1

    tasks_by_split: dict[str, list[V2RequirementTask]] = {
        split: [] for split in split_counts
    }
    for index, spec in enumerate(specs):
        audit = _audit_for_spec(index)
        for missing_rate in (0.3, 0.5, 0.7):
            tasks_by_split[spec.split].append(
                _mask_v2_spec(spec, missing_rate, rng, audit)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "complete_specs.jsonl", specs)
    for split, tasks in tasks_by_split.items():
        _write_jsonl(output_dir / f"{split}_tasks.jsonl", tasks)
    manifest = DatasetManifest(
        complete_spec_count=len(specs),
        task_count=sum(len(tasks) for tasks in tasks_by_split.values()),
        split_spec_counts={split: len(items) for split, items in split_specs.items()},
        train_spec_ids=[item.spec_id for item in split_specs["train"]],
        validation_spec_ids=[item.spec_id for item in split_specs["validation"]],
        test_spec_ids=[item.spec_id for item in split_specs["test"]],
        seed=seed,
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest
