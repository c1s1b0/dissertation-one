from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import (
    CompleteSpec,
    DatasetManifest,
    RequirementTask,
    generate_complete_spec,
)
from paper1_active_req.relations.config import (
    V3RelationConfig,
    load_v3_relation_config,
)
from paper1_active_req.solver.engine import ConstraintEngine


V3Mechanism = Literal[
    "interface_compatibility",
    "continuous_clearance",
    "conditional_tolerance",
]

_MASTER_SEED = 20260806
_MISSING_RATES = (0.3, 0.5, 0.7)
_DEVELOPMENT_TRAIN_SPEC_COUNT = 180
_DEVELOPMENT_VALIDATION_SPEC_COUNT = 60
_TEST_SPECS_PER_MECHANISM = 20
_TEST_PIVOT_SPECS_PER_MECHANISM = 15
_MECHANISMS: tuple[V3Mechanism, ...] = (
    "interface_compatibility",
    "continuous_clearance",
    "conditional_tolerance",
)


class V3AuditMetadata(BaseModel):
    module_id: str
    pivot_field_id: str | None
    affected_critical_field_ids: tuple[str, ...] = ()
    mechanism: V3Mechanism


class V3RequirementTask(RequirementTask):
    audit: V3AuditMetadata

    def ground_truth_to_requirement_state(self) -> RequirementState:
        return RequirementState(values=self.ground_truth).model_copy(deep=True)


class V3DatasetManifest(DatasetManifest):
    schema_hash: str
    relation_config_hash: str
    freeze_manifest_hash: str | None = None


class V3TestReleaseManifest(V3DatasetManifest):
    test_tasks_hash: str
    test_complete_specs_hash: str
    released_task_ids: list[str]
    released_spec_ids: list[str]


def stable_child_seed(master_seed: int, split: str) -> int:
    payload = f"v3:{master_seed}:{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_master_seed(master_seed: int) -> None:
    if master_seed != _MASTER_SEED:
        raise ValueError(f"v3 master seed must be {_MASTER_SEED}")


def _load_generation_context(
    schema_path: Path,
    relation_config_path: Path,
) -> tuple[dict[str, FieldDefinition], ConstraintEngine, V3RelationConfig]:
    schema = load_field_schema(schema_path)
    relation_config = load_v3_relation_config(relation_config_path)
    return schema, ConstraintEngine(schema), relation_config


def _observed(value: object, source: str) -> FieldValue:
    return FieldValue(
        value=value,
        status=FieldStatus.OBSERVED,
        source=source,
    )


def _truncated_noise(
    rng: np.random.Generator,
    *,
    standard_deviation: float,
    limit: float,
) -> float:
    return float(
        np.clip(rng.normal(0.0, standard_deviation), -limit, limit)
    )


def _audit_for(
    mechanism: V3Mechanism,
    *,
    pivot: bool,
) -> V3AuditMetadata:
    if mechanism == "interface_compatibility":
        pivot_field_id = "upstream_output_interface"
        affected_ids = ("station_input_interface",)
    elif mechanism == "continuous_clearance":
        pivot_field_id = "side_clearance_mm"
        affected_ids = ("available_width_mm",)
    else:
        pivot_field_id = "workpiece_fragile"
        affected_ids = (
            "positioning_accuracy_mm",
            "repeatability_mm",
        )
    return V3AuditMetadata(
        module_id=f"v3.{mechanism}",
        pivot_field_id=pivot_field_id if pivot else None,
        affected_critical_field_ids=affected_ids if pivot else (),
        mechanism=mechanism,
    )


def _apply_interface_compatibility(
    values: dict[str, FieldValue],
    rng: np.random.Generator,
    schema: dict[str, FieldDefinition],
) -> None:
    valid_categories = schema["upstream_output_interface"].categories
    original = str(values["upstream_output_interface"].value)
    alternatives = [category for category in valid_categories if category != original]
    selected = str(rng.choice(alternatives or valid_categories))
    values["upstream_output_interface"] = _observed(
        selected, "v3_interface_perturbation"
    )
    values["station_input_interface"] = _observed(
        selected, "v3_interface_perturbation"
    )


def _apply_continuous_clearance(
    values: dict[str, FieldValue],
    rng: np.random.Generator,
    relation_config: V3RelationConfig,
) -> None:
    width = float(values["workpiece_width_mm"].value)
    side_clearance = float(values["side_clearance_mm"].value)
    tolerance = relation_config.engineering_tolerances["available_width_mm"]
    coefficient = float(rng.uniform(0.015, 0.045))
    nonlinear_excess = coefficient * (width / 100.0) ** 2
    noise = _truncated_noise(
        rng,
        standard_deviation=tolerance * 0.15,
        limit=tolerance * 0.30,
    )
    available_width = width + 2.0 * side_clearance + tolerance + nonlinear_excess + noise
    values["available_width_mm"] = _observed(
        float(available_width), "v3_continuous_clearance"
    )


def _apply_conditional_tolerance(
    values: dict[str, FieldValue],
    rng: np.random.Generator,
    relation_config: V3RelationConfig,
) -> None:
    fragile = bool(values["workpiece_fragile"].value)
    repeatability = float(rng.uniform(0.02, 2.0))
    tolerance = relation_config.engineering_tolerances["positioning_accuracy_mm"]
    linear_margin = 0.12 if fragile else 0.42
    curvature = 0.08 if fragile else 0.03
    bounded_noise = _truncated_noise(
        rng,
        standard_deviation=tolerance * 0.25,
        limit=tolerance * 0.50,
    )
    positioning_accuracy = (
        repeatability
        + linear_margin
        + curvature * repeatability**2
        + bounded_noise
    )
    values["repeatability_mm"] = _observed(
        repeatability, "v3_conditional_tolerance"
    )
    values["positioning_accuracy_mm"] = _observed(
        float(positioning_accuracy), "v3_conditional_tolerance"
    )


def _apply_mechanism(
    spec: CompleteSpec,
    audit: V3AuditMetadata,
    rng: np.random.Generator,
    schema: dict[str, FieldDefinition],
    relation_config: V3RelationConfig,
    engine: ConstraintEngine,
) -> CompleteSpec:
    if audit.pivot_field_id is None:
        return spec

    values = {
        field_id: field_value.model_copy(deep=True)
        for field_id, field_value in spec.values.items()
    }
    if audit.mechanism == "interface_compatibility":
        _apply_interface_compatibility(values, rng, schema)
    elif audit.mechanism == "continuous_clearance":
        _apply_continuous_clearance(values, rng, relation_config)
    else:
        _apply_conditional_tolerance(values, rng, relation_config)

    generated = spec.model_copy(update={"values": values})
    if engine.check_satisfiability(generated.to_requirement_state()).satisfiable is not True:
        raise RuntimeError(f"v3 mechanism generated an unsatisfiable spec: {spec.spec_id}")
    return generated


def _mask_spec(
    spec: CompleteSpec,
    audit: V3AuditMetadata,
    missing_rate: float,
    rng: np.random.Generator,
    queryable_field_ids: set[str],
) -> V3RequirementTask:
    hidden_count = round(len(queryable_field_ids) * missing_rate)
    required_hidden_ids = {
        field_id
        for field_id in (
            audit.pivot_field_id,
            *audit.affected_critical_field_ids,
        )
        if field_id is not None
    }
    if not required_hidden_ids.issubset(queryable_field_ids):
        raise ValueError("v3 pivot fields must be queryable")
    if len(required_hidden_ids) > hidden_count:
        raise ValueError("v3 missing rate cannot hide all required pivot fields")

    remaining_ids = sorted(queryable_field_ids - required_hidden_ids)
    extra_count = hidden_count - len(required_hidden_ids)
    extra_hidden_ids = set(
        str(field_id)
        for field_id in np.atleast_1d(
            rng.choice(remaining_ids, size=extra_count, replace=False)
        )
    )
    hidden_ids = required_hidden_ids | extra_hidden_ids
    initial_values = {
        field_id: field_value.model_copy(deep=True)
        for field_id, field_value in spec.values.items()
    }
    for field_id in hidden_ids:
        initial_values[field_id] = FieldValue(
            value=None,
            status=FieldStatus.HIDDEN,
            source="v3_mask",
        )
    return V3RequirementTask(
        task_id=f"{spec.spec_id}__m{int(round(missing_rate * 100))}",
        spec_id=spec.spec_id,
        split=spec.split,
        missing_rate=missing_rate,
        initial_state=RequirementState(values=initial_values),
        ground_truth={
            field_id: field_value.model_copy(deep=True)
            for field_id, field_value in spec.values.items()
        },
        audit=audit,
    )


def _write_jsonl(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(item.model_dump_json() for item in items)
    path.write_text(f"{text}\n" if text else "", encoding="utf-8")


def _generate_specs_and_tasks(
    *,
    master_seed: int,
    split: str,
    spec_definitions: list[tuple[str, V3AuditMetadata]],
    schema: dict[str, FieldDefinition],
    engine: ConstraintEngine,
    relation_config: V3RelationConfig,
) -> tuple[list[CompleteSpec], list[V3RequirementTask]]:
    rng = np.random.default_rng(stable_child_seed(master_seed, split))
    queryable_field_ids = {
        field_id for field_id, definition in schema.items() if definition.queryable
    }
    specs: list[CompleteSpec] = []
    tasks: list[V3RequirementTask] = []
    for spec_id, audit in spec_definitions:
        spec = generate_complete_spec(rng, schema, engine, spec_id).model_copy(
            update={"split": split}
        )
        generated = _apply_mechanism(
            spec,
            audit,
            rng,
            schema,
            relation_config,
            engine,
        )
        specs.append(generated)
        tasks.extend(
            _mask_spec(
                generated,
                audit,
                missing_rate,
                rng,
                queryable_field_ids,
            )
            for missing_rate in _MISSING_RATES
        )
    return specs, tasks


def _manifest(
    *,
    specs_by_split: dict[str, list[CompleteSpec]],
    tasks_by_split: dict[str, list[V3RequirementTask]],
    master_seed: int,
    schema_path: Path,
    relation_config_path: Path,
    freeze_manifest_hash: str | None = None,
) -> V3DatasetManifest:
    return V3DatasetManifest(
        complete_spec_count=sum(len(items) for items in specs_by_split.values()),
        task_count=sum(len(items) for items in tasks_by_split.values()),
        split_spec_counts={
            split: len(items) for split, items in specs_by_split.items()
        },
        train_spec_ids=[item.spec_id for item in specs_by_split.get("train", [])],
        validation_spec_ids=[
            item.spec_id for item in specs_by_split.get("validation", [])
        ],
        test_spec_ids=[item.spec_id for item in specs_by_split.get("test", [])],
        seed=master_seed,
        schema_hash=_hash_file(schema_path),
        relation_config_hash=_hash_file(relation_config_path),
        freeze_manifest_hash=freeze_manifest_hash,
    )


def generate_v3_development_dataset(
    master_seed: int,
    output_dir: Path,
    schema_path: Path,
    relation_config_path: Path,
) -> V3DatasetManifest:
    from paper1_active_req.experiment.v3_paths import validate_v3_input_paths, validate_v3_output_path
    validate_v3_input_paths(schema_path, relation_config_path)
    validate_v3_output_path(output_dir)
    _require_master_seed(master_seed)
    schema, engine, relation_config = _load_generation_context(
        schema_path, relation_config_path
    )
    specs_by_split: dict[str, list[CompleteSpec]] = {}
    tasks_by_split: dict[str, list[V3RequirementTask]] = {}
    for split, count in (
        ("train", _DEVELOPMENT_TRAIN_SPEC_COUNT),
        ("validation", _DEVELOPMENT_VALIDATION_SPEC_COUNT),
    ):
        definitions = [
            (
                f"v3-{split}-{index:04d}",
                _audit_for(_MECHANISMS[index % len(_MECHANISMS)], pivot=True),
            )
            for index in range(1, count + 1)
        ]
        specs_by_split[split], tasks_by_split[split] = _generate_specs_and_tasks(
            master_seed=master_seed,
            split=split,
            spec_definitions=definitions,
            schema=schema,
            engine=engine,
            relation_config=relation_config,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train_complete_specs.jsonl", specs_by_split["train"])
    _write_jsonl(
        output_dir / "validation_complete_specs.jsonl", specs_by_split["validation"]
    )
    _write_jsonl(output_dir / "train_tasks.jsonl", tasks_by_split["train"])
    _write_jsonl(output_dir / "validation_tasks.jsonl", tasks_by_split["validation"])
    manifest = _manifest(
        specs_by_split=specs_by_split,
        tasks_by_split=tasks_by_split,
        master_seed=master_seed,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    (output_dir / "development_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def _validated_frozen_manifest(
    *,
    master_seed: int,
    schema_path: Path,
    relation_config_path: Path,
    freeze_manifest_path: Path,
) -> str:
    from paper1_active_req.experiment.v3_runner import V3FreezeManifest, _verify_freeze_manifest
    try:
        raw = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen manifest is required") from exc
    if not isinstance(raw, dict):
        raise ValueError("frozen manifest must be a JSON object")

    if set(raw) != set(V3FreezeManifest.model_fields):
        raise ValueError("frozen manifest is incomplete")
    try:
        manifest = _verify_freeze_manifest(freeze_manifest_path)
    except ValueError as error:
        raise ValueError("frozen manifest is incomplete") from error
    if raw["status"] != "frozen" or raw["master_seed"] != master_seed:
        raise ValueError("frozen manifest does not match the release")
    if raw["master_seed"] != _MASTER_SEED:
        raise ValueError("frozen manifest has the wrong master seed")
    if manifest.schema_hash != _hash_file(schema_path):
        raise ValueError("frozen manifest schema hash does not match")
    if manifest.relation_config_hash != _hash_file(relation_config_path):
        raise ValueError("frozen manifest relation config hash does not match")
    return _hash_file(freeze_manifest_path)


def release_v3_test_dataset(
    master_seed: int,
    output_dir: Path,
    schema_path: Path,
    relation_config_path: Path,
    freeze_manifest_path: Path,
) -> V3DatasetManifest:
    from paper1_active_req.experiment.v3_paths import validate_v3_input_paths, validate_v3_output_path
    validate_v3_input_paths(schema_path, relation_config_path, freeze_manifest_path)
    validate_v3_output_path(output_dir)
    _require_master_seed(master_seed)
    frozen_manifest_hash = _validated_frozen_manifest(
        master_seed=master_seed,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    protected_paths = (
        output_dir / "test_complete_specs.jsonl",
        output_dir / "test_tasks.jsonl",
        output_dir / "test_release_manifest.json",
    )
    existing_path = next((path for path in protected_paths if path.exists()), None)
    if existing_path is not None:
        raise FileExistsError(f"refusing to overwrite {existing_path.name}")

    schema, engine, relation_config = _load_generation_context(
        schema_path, relation_config_path
    )
    definitions = [
        (
            f"v3-test-{mechanism}-{index:03d}",
            _audit_for(
                mechanism,
                pivot=index <= _TEST_PIVOT_SPECS_PER_MECHANISM,
            ),
        )
        for mechanism in _MECHANISMS
        for index in range(1, _TEST_SPECS_PER_MECHANISM + 1)
    ]
    test_specs, test_tasks = _generate_specs_and_tasks(
        master_seed=master_seed,
        split="test",
        spec_definitions=definitions,
        schema=schema,
        engine=engine,
        relation_config=relation_config,
    )
    specs_by_split = {"test": test_specs}
    tasks_by_split = {"test": test_tasks}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "test_complete_specs.jsonl", test_specs)
    _write_jsonl(output_dir / "test_tasks.jsonl", test_tasks)
    base_manifest = _manifest(
        specs_by_split=specs_by_split,
        tasks_by_split=tasks_by_split,
        master_seed=master_seed,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        freeze_manifest_hash=frozen_manifest_hash,
    )
    manifest = V3TestReleaseManifest(
        **base_manifest.model_dump(),
        test_tasks_hash=_hash_file(output_dir / "test_tasks.jsonl"),
        test_complete_specs_hash=_hash_file(output_dir / "test_complete_specs.jsonl"),
        released_task_ids=[task.task_id for task in test_tasks],
        released_spec_ids=[spec.spec_id for spec in test_specs],
    )
    (output_dir / "test_release_manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest
