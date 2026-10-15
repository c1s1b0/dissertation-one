import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

import paper1_active_req.experiment.v3_runner as v3_runner
import paper1_active_req.generator.v3_relational_specs as v3_generator
from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.experiment.v3_runner import freeze_v3_artifacts
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import (
    CompleteSpec,
    generate_complete_spec,
)
from paper1_active_req.generator.v3_relational_specs import (
    V3AuditMetadata,
    V3RequirementTask,
    generate_v3_development_dataset,
    release_v3_test_dataset,
    stable_child_seed,
)
from paper1_active_req.solver.engine import ConstraintEngine


MASTER_SEED = 20260806
SCHEMA_PATH = Path("configs/fields.yaml")
RELATION_CONFIG_PATH = Path("configs/v3_relations.yaml")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_inputs(path: Path) -> dict[str, Path]:
    path.mkdir(parents=True)
    schema = load_field_schema(SCHEMA_PATH)
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(31)
    base_training = [
        generate_complete_spec(rng, schema, engine, f"train-{index}").model_copy(
            update={"split": "train"}
        )
        for index in range(3)
    ]
    training = [
        base_training[index % len(base_training)].model_copy(
            update={"spec_id": f"train-{index:03d}"}
        )
        for index in range(180)
    ]
    training_path = path / "train_complete_specs.jsonl"
    training_path.write_text(
        "\n".join(spec.model_dump_json() for spec in training) + "\n",
        encoding="utf-8",
    )

    values = {
        field_id: value.model_copy(deep=True)
        for field_id, value in training[0].values.items()
    }
    values["workpiece_mass_kg"] = FieldValue(
        value=None,
        status=FieldStatus.HIDDEN,
        source="validation",
    )
    validation_tasks = [
        V3RequirementTask(
            task_id=f"validation-{index:02d}-m{int(rate * 100)}",
            spec_id=f"validation-{index:02d}",
            split="validation",
            missing_rate=rate,
            initial_state=RequirementState(
                values={
                    key: value.model_copy(deep=True)
                    for key, value in values.items()
                }
            ),
            ground_truth={
                key: value.model_copy(deep=True)
                for key, value in training[0].values.items()
            },
            audit=V3AuditMetadata(
                module_id="v3.interface_compatibility",
                pivot_field_id=None,
                mechanism="interface_compatibility",
            ),
        )
        for index in range(60)
        for rate in (0.3, 0.5, 0.7)
    ]
    validation_path = path / "validation_tasks.jsonl"
    validation_path.write_text(
        "\n".join(task.model_dump_json() for task in validation_tasks) + "\n",
        encoding="utf-8",
    )

    development_manifest = path / "development_manifest.json"
    development_manifest.write_text(
        json.dumps(
            {
                "complete_spec_count": 240,
                "task_count": 720,
                "split_spec_counts": {"train": 180, "validation": 60},
                "train_spec_ids": [spec.spec_id for spec in training],
                "validation_spec_ids": [
                    f"validation-{index:02d}" for index in range(60)
                ],
                "test_spec_ids": [],
                "seed": MASTER_SEED,
                "schema_hash": _file_hash(SCHEMA_PATH),
                "relation_config_hash": _file_hash(RELATION_CONFIG_PATH),
            }
        ),
        encoding="utf-8",
    )
    experiment_config = path / "experiment.yaml"
    experiment_config.write_text(
        "master_seed: 20260806\n"
        "budget: 8\n"
        "lambda_candidates: [0.0, 0.05, 0.1, 0.2]\n",
        encoding="utf-8",
    )

    model_hash = v3_runner._json_hash(
        [spec.model_dump(mode="json") for spec in training]
    )
    candidates = []
    for lambda_cost in (0.0, 0.05, 0.1, 0.2):
        config_hash = v3_runner._json_hash(
            {
                "schema_hash": _file_hash(SCHEMA_PATH),
                "relation_config_hash": _file_hash(RELATION_CONFIG_PATH),
                "seed": MASTER_SEED,
                "budget": 8,
                "lambda_cost": lambda_cost,
                "policy_names": ["tr_mla"],
            }
        )
        candidates.append(
            {
                "lambda_cost": lambda_cost,
                "policy_names": ["tr_mla"],
                "records": [
                    {
                        "task_id": task.task_id,
                        "spec_id": task.spec_id,
                        "split": "validation",
                        "missing_rate": task.missing_rate,
                        "policy_name": "tr_mla",
                        "config_hash": config_hash,
                        "model_hash": model_hash,
                        "metrics": {
                            "crr_at_8": 0.5,
                            "crr_auc": 0.5,
                            "answer_cost": 1.0,
                        },
                    }
                    for task in validation_tasks
                ],
            }
        )
    selection = path / "selected_lambda.json"
    selection.write_text(
        json.dumps(
            {
                "selection_split": "validation",
                "seed": MASTER_SEED,
                "budget": 8,
                "selected_lambda": {"lambda_cost": 0.0},
                "selection_metrics": {
                    "crr_at_8": 0.5,
                    "crr_auc": 0.5,
                    "answer_cost": 1.0,
                },
                "model_summary_hash": v3_runner._json_hash(
                    sorted([model_hash] * 720)
                ),
                "validation_tasks_hash": _file_hash(validation_path),
                "training_specs_hash": _file_hash(training_path),
                "schema_hash": _file_hash(SCHEMA_PATH),
                "relation_config_hash": _file_hash(RELATION_CONFIG_PATH),
                "experiment_config_hash": _file_hash(experiment_config),
                "candidate_records_hash": v3_runner._json_hash(candidates),
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    return {
        "development_manifest": development_manifest,
        "selection": selection,
        "experiment_config": experiment_config,
    }


@pytest.fixture(scope="session")
def valid_frozen_manifest(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v3-release-freeze")
    inputs = _freeze_inputs(path / "inputs")
    freeze_path = path / "frozen_manifest.json"
    freeze_v3_artifacts(
        development_manifest_path=inputs["development_manifest"],
        selection_file=inputs["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=inputs["experiment_config"],
        source_paths=v3_runner.formal_v3_source_paths(),
        output_path=freeze_path,
    )
    return freeze_path


def _frozen_manifest(
    path: Path,
    baseline: Path,
    overrides: dict[str, object] | None = None,
) -> Path:
    freeze_path = path / "frozen_manifest.json"
    content: dict[str, object] = json.loads(baseline.read_text(encoding="utf-8"))
    if overrides is not None:
        content.update(overrides)
    freeze_path.write_text(
        json.dumps(content, indent=2),
        encoding="utf-8",
    )
    return freeze_path


def _read_tasks(path: Path) -> list[V3RequirementTask]:
    return [
        V3RequirementTask.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _observed(value: object) -> FieldValue:
    return FieldValue(
        value=value,
        status=FieldStatus.OBSERVED,
        source="test_fixture",
    )


def _hidden_count(task: V3RequirementTask) -> int:
    return sum(
        value.status == FieldStatus.HIDDEN
        for value in task.initial_state.values.values()
    )


def test_stable_child_seed_is_reproducible_and_split_specific():
    assert stable_child_seed(MASTER_SEED, "train") == stable_child_seed(
        MASTER_SEED, "train"
    )
    assert len(
        {
            stable_child_seed(MASTER_SEED, split)
            for split in ("train", "validation", "test")
        }
    ) == 3


def test_v3_development_generation_does_not_create_test_artifacts(
    tmp_path: Path,
):
    output_dir = tmp_path / "v3"
    manifest = generate_v3_development_dataset(
        master_seed=MASTER_SEED,
        output_dir=output_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
    )
    train_tasks = _read_tasks(output_dir / "train_tasks.jsonl")
    validation_tasks = _read_tasks(output_dir / "validation_tasks.jsonl")
    tasks = [*train_tasks, *validation_tasks]
    tasks_by_spec: dict[str, list[V3RequirementTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_spec[task.spec_id].append(task)

    assert manifest.split_spec_counts == {"train": 180, "validation": 60}
    assert manifest.task_count == 720
    assert set(manifest.train_spec_ids).isdisjoint(manifest.validation_spec_ids)
    assert len(train_tasks) == 540
    assert len(validation_tasks) == 180
    assert Counter(task.missing_rate for task in tasks) == {
        0.3: 240,
        0.5: 240,
        0.7: 240,
    }
    assert all(
        {task.missing_rate for task in task_group} == {0.3, 0.5, 0.7}
        for task_group in tasks_by_spec.values()
    )
    assert all(
        _hidden_count(task) == {0.3: 7, 0.5: 12, 0.7: 16}[task.missing_rate]
        for task in tasks
    )
    assert not any(
        (output_dir / filename).exists()
        for filename in (
            "test_complete_specs.jsonl",
            "test_tasks.jsonl",
            "test_release_manifest.json",
        )
    )
    assert not any(spec_id.startswith("v2-") for spec_id in manifest.train_spec_ids)


def test_private_generation_protocol_counts_drive_real_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_frozen_manifest: Path,
):
    """缩小私有协议仍必须经过真实 development/release 生成循环。"""
    monkeypatch.setattr(
        v3_generator, "_DEVELOPMENT_TRAIN_SPEC_COUNT", 12, raising=False
    )
    monkeypatch.setattr(
        v3_generator, "_DEVELOPMENT_VALIDATION_SPEC_COUNT", 6, raising=False
    )
    monkeypatch.setattr(
        v3_generator, "_TEST_SPECS_PER_MECHANISM", 2, raising=False
    )
    monkeypatch.setattr(
        v3_generator, "_TEST_PIVOT_SPECS_PER_MECHANISM", 1, raising=False
    )

    output_dir = tmp_path / "v3"
    development = generate_v3_development_dataset(
        master_seed=MASTER_SEED,
        output_dir=output_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
    )
    release = release_v3_test_dataset(
        master_seed=MASTER_SEED,
        output_dir=output_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        freeze_manifest_path=_frozen_manifest(tmp_path, valid_frozen_manifest),
    )
    test_tasks = _read_tasks(output_dir / "test_tasks.jsonl")
    audit_by_spec = {task.spec_id: task.audit for task in test_tasks}

    assert development.split_spec_counts == {"train": 12, "validation": 6}
    assert release.split_spec_counts == {"test": 6}
    assert Counter(audit.mechanism for audit in audit_by_spec.values()) == {
        "interface_compatibility": 2,
        "continuous_clearance": 2,
        "conditional_tolerance": 2,
    }
    assert Counter(
        (audit.mechanism, audit.pivot_field_id is not None)
        for audit in audit_by_spec.values()
    ) == {
        ("interface_compatibility", True): 1,
        ("interface_compatibility", False): 1,
        ("continuous_clearance", True): 1,
        ("continuous_clearance", False): 1,
        ("conditional_tolerance", True): 1,
        ("conditional_tolerance", False): 1,
    }


def test_test_release_requires_frozen_manifest(tmp_path: Path):
    with pytest.raises(ValueError, match="frozen manifest"):
        release_v3_test_dataset(
            master_seed=MASTER_SEED,
            output_dir=tmp_path / "v3",
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            freeze_manifest_path=tmp_path / "missing.json",
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"status": "draft"},
        {"master_seed": MASTER_SEED + 1},
        {"schema_hash": "not-the-schema"},
        {"relation_config_hash": "not-the-relation-config"},
        {"experiment_config_hash": ""},
        {"source_hash": ""},
        {"model_summary_hash": ""},
        {"selected_lambda": ""},
        {"selection_file_hash": ""},
    ),
    ids=(
        "status",
        "master_seed",
        "schema_hash",
        "relation_config_hash",
        "experiment_config_hash",
        "source_hash",
        "model_summary_hash",
        "selected_lambda",
        "selection_file_hash",
    ),
)
def test_test_release_rejects_invalid_frozen_manifest_fields(
    tmp_path: Path,
    valid_frozen_manifest: Path,
    overrides: dict[str, object],
):
    with pytest.raises(ValueError, match="frozen manifest"):
        release_v3_test_dataset(
            master_seed=MASTER_SEED,
            output_dir=tmp_path / "v3",
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            freeze_manifest_path=_frozen_manifest(
                tmp_path,
                valid_frozen_manifest,
                overrides,
            ),
        )


def test_test_release_writes_balanced_audited_satisfiable_tasks(
    tmp_path: Path,
    valid_frozen_manifest: Path,
):
    output_dir = tmp_path / "v3"
    manifest = release_v3_test_dataset(
        master_seed=MASTER_SEED,
        output_dir=output_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        freeze_manifest_path=_frozen_manifest(tmp_path, valid_frozen_manifest),
    )

    tasks = _read_tasks(output_dir / "test_tasks.jsonl")
    specs = [
        CompleteSpec.model_validate_json(line)
        for line in (output_dir / "test_complete_specs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    tasks_by_spec: dict[str, list[V3RequirementTask]] = {}
    for task in tasks:
        tasks_by_spec.setdefault(task.spec_id, []).append(task)
    specs_by_id = {spec.spec_id: spec for spec in specs}

    assert manifest.split_spec_counts == {"test": 60}
    assert manifest.complete_spec_count == 60
    assert manifest.task_count == 180
    assert len(specs) == 60
    assert len(tasks) == 180
    assert {
        mechanism: sum(
            task.audit.mechanism == mechanism
            for task in tasks_by_spec.values()
            for task in task[:1]
        )
        for mechanism in (
            "interface_compatibility",
            "continuous_clearance",
            "conditional_tolerance",
        )
    } == {
        "interface_compatibility": 20,
        "continuous_clearance": 20,
        "conditional_tolerance": 20,
    }

    pivot_spec_counts = {
        mechanism: sum(
            task_group[0].audit.pivot_field_id is not None
            for task_group in tasks_by_spec.values()
            if task_group[0].audit.mechanism == mechanism
        )
        for mechanism in (
            "interface_compatibility",
            "continuous_clearance",
            "conditional_tolerance",
        )
    }
    assert pivot_spec_counts == {
        "interface_compatibility": 15,
        "continuous_clearance": 15,
        "conditional_tolerance": 15,
    }

    for task_group in tasks_by_spec.values():
        audit = task_group[0].audit
        assert len(task_group) == 3
        if audit.pivot_field_id is None:
            assert audit.affected_critical_field_ids == ()
            continue
        for task in task_group:
            assert task.initial_state.values[audit.pivot_field_id].status == FieldStatus.HIDDEN
            assert all(
                task.initial_state.values[field_id].status == FieldStatus.HIDDEN
                for field_id in audit.affected_critical_field_ids
            )

    schema = load_field_schema(SCHEMA_PATH)
    baseline_first_spec = generate_complete_spec(
        np.random.default_rng(stable_child_seed(MASTER_SEED, "test")),
        schema,
        ConstraintEngine(schema),
        "v3-test-interface_compatibility-001",
    )
    released_first_spec = specs_by_id["v3-test-interface_compatibility-001"]
    selected_interface = released_first_spec.values[
        "upstream_output_interface"
    ].value
    assert selected_interface in schema["upstream_output_interface"].categories
    assert selected_interface != baseline_first_spec.values[
        "upstream_output_interface"
    ].value
    assert released_first_spec.values["station_input_interface"].value == selected_interface

    continuous_excesses: list[float] = []
    for spec_id, task_group in tasks_by_spec.items():
        audit = task_group[0].audit
        if audit.pivot_field_id is None:
            continue
        values = specs_by_id[spec_id].values
        if audit.mechanism == "continuous_clearance":
            width = float(values["workpiece_width_mm"].value)
            side_clearance = float(values["side_clearance_mm"].value)
            excess = float(values["available_width_mm"].value) - width - 2.0 * side_clearance
            assert 7.0 + 0.015 * (width / 100.0) ** 2 <= excess
            assert excess <= 13.0 + 0.045 * (width / 100.0) ** 2
            continuous_excesses.append(excess)
        elif audit.mechanism == "conditional_tolerance":
            repeatability = float(values["repeatability_mm"].value)
            margin = float(values["positioning_accuracy_mm"].value) - repeatability
            if bool(values["workpiece_fragile"].value):
                assert 0.095 + 0.08 * repeatability**2 <= margin
                assert margin <= 0.145 + 0.08 * repeatability**2
            else:
                assert 0.395 + 0.03 * repeatability**2 <= margin
                assert margin <= 0.445 + 0.03 * repeatability**2
    assert any(excess > 13.0 for excess in continuous_excesses)

    engine = ConstraintEngine(schema)
    assert all(
        engine.check_satisfiability(spec.to_requirement_state()).satisfiable
        for spec in specs
    )


def test_test_release_refuses_to_overwrite_existing_tasks(
    tmp_path: Path,
    valid_frozen_manifest: Path,
):
    output_dir = tmp_path / "v3"
    output_dir.mkdir()
    (output_dir / "test_tasks.jsonl").write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="test_tasks.jsonl"):
        release_v3_test_dataset(
            master_seed=MASTER_SEED,
            output_dir=output_dir,
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            freeze_manifest_path=_frozen_manifest(tmp_path, valid_frozen_manifest),
        )


@pytest.mark.parametrize(
    "filename",
    (
        "test_complete_specs.jsonl",
        "test_tasks.jsonl",
        "test_release_manifest.json",
    ),
)
def test_test_release_refuses_to_overwrite_any_test_artifact(
    tmp_path: Path,
    valid_frozen_manifest: Path,
    filename: str,
):
    output_dir = tmp_path / "v3"
    output_dir.mkdir()
    (output_dir / filename).write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match=filename):
        release_v3_test_dataset(
            master_seed=MASTER_SEED,
            output_dir=output_dir,
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            freeze_manifest_path=_frozen_manifest(tmp_path, valid_frozen_manifest),
        )


def test_test_release_is_byte_reproducible_across_distinct_tmp_outputs(
    tmp_path: Path,
    valid_frozen_manifest: Path,
):
    freeze_manifest_path = _frozen_manifest(tmp_path, valid_frozen_manifest)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    release_v3_test_dataset(
        master_seed=MASTER_SEED,
        output_dir=first_output,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        freeze_manifest_path=freeze_manifest_path,
    )
    release_v3_test_dataset(
        master_seed=MASTER_SEED,
        output_dir=second_output,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        freeze_manifest_path=freeze_manifest_path,
    )

    for filename in (
        "test_complete_specs.jsonl",
        "test_tasks.jsonl",
        "test_release_manifest.json",
    ):
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()


def test_ground_truth_to_requirement_state_is_deeply_isolated():
    task = V3RequirementTask(
        task_id="task",
        spec_id="spec",
        split="test",
        missing_rate=0.3,
        initial_state=RequirementState(
            values={"workpiece_fragile": _observed(True)}
        ),
        ground_truth={"workpiece_fragile": _observed(True)},
        audit=V3AuditMetadata(
            module_id="v3.conditional_tolerance",
            pivot_field_id="workpiece_fragile",
            affected_critical_field_ids=(
                "positioning_accuracy_mm",
                "repeatability_mm",
            ),
            mechanism="conditional_tolerance",
        ),
    )

    copied_state = task.ground_truth_to_requirement_state()
    copied_state.values["workpiece_fragile"].value = False
    copied_state.asked_fields.append("workpiece_fragile")

    assert task.ground_truth["workpiece_fragile"].value is True
    assert task.ground_truth["workpiece_fragile"] is not copied_state.values[
        "workpiece_fragile"
    ]
    assert task.ground_truth_to_requirement_state().asked_fields == []
