import json
from pathlib import Path

import pytest

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.v2_pivotal_specs import (
    V2RequirementTask,
    generate_v2_dataset,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _load_tasks(path: Path) -> list[V2RequirementTask]:
    return [
        V2RequirementTask.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def v2_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成一次固定数据；各断言只读取同一份不可变测试产物。"""
    output_dir = tmp_path_factory.mktemp("v2")
    generate_v2_dataset(
        seed=20260805,
        output_dir=output_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    return output_dir


def test_v2_dataset_keeps_splits_disjoint_and_balances_test_mechanisms(
    v2_dataset_dir: Path,
):
    """防止生成器把规格跨 split 复用或漏掉预注册的三种主机制。"""
    output_dir = v2_dataset_dir
    manifest_payload = json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest_payload["complete_spec_count"] == 180
    assert manifest_payload["task_count"] == 540
    assert manifest_payload["split_spec_counts"] == {
        "train": 108,
        "validation": 36,
        "test": 36,
    }
    assert set(manifest_payload["train_spec_ids"]).isdisjoint(manifest_payload["validation_spec_ids"])
    assert set(manifest_payload["train_spec_ids"]).isdisjoint(manifest_payload["test_spec_ids"])
    assert set(manifest_payload["validation_spec_ids"]).isdisjoint(manifest_payload["test_spec_ids"])

    test_tasks = _load_tasks(output_dir / "test_tasks.jsonl")
    assert len(test_tasks) == 108
    assert {task.spec_id for task in test_tasks} == set(manifest_payload["test_spec_ids"])
    mechanism_spec_counts = {
        mechanism: {
            task.spec_id
            for task in test_tasks
            if task.audit.mechanism == mechanism
        }
        for mechanism in {
            "interface_unlock",
            "interval_reduction",
            "conditional_correlation",
        }
    }
    assert {name: len(ids) for name, ids in mechanism_spec_counts.items()} == {
        "interface_unlock": 12,
        "interval_reduction": 12,
        "conditional_correlation": 12,
    }
    assert any(task.audit.pivot_field_id is None for task in test_tasks)
    assert all(task.spec_id.startswith("v2-spec-") for task in test_tasks)


def test_v2_complete_specs_are_satisfiable_and_pivots_are_lower_criticality(
    v2_dataset_dir: Path,
):
    """防止不可满足规格或高关键字段伪装为枢纽造成捷径。"""
    output_dir = v2_dataset_dir
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    tasks = _load_tasks(output_dir / "test_tasks.jsonl")

    assert json.loads(
        (output_dir / "manifest.json").read_text(encoding="utf-8")
    )["seed"] == 20260805
    assert all(
        engine.check(task.ground_truth_to_requirement_state()).satisfiable
        for task in tasks
    )
    for task in tasks:
        pivot = task.audit.pivot_field_id
        if pivot is None:
            assert task.audit.affected_critical_field_ids == ()
            continue
        assert all(
            schema[pivot].criticality < schema[field_id].criticality
            for field_id in task.audit.affected_critical_field_ids
        )


def test_v2_pivotal_mechanisms_encode_learnable_conditional_relations(
    v2_dataset_dir: Path,
):
    """防止 audit 只是标签而不对应可由训练数据学习的工程关系。"""
    output_dir = v2_dataset_dir
    tasks = _load_tasks(output_dir / "train_tasks.jsonl")
    representative = {
        task.audit.mechanism: task
        for task in tasks
        if task.audit.pivot_field_id is not None
    }

    interface_values = representative["interface_unlock"].ground_truth
    assert (
        interface_values["upstream_output_interface"].value
        == interface_values["station_input_interface"].value
    )

    interval_values = representative["interval_reduction"].ground_truth
    assert float(interval_values["available_width_mm"].value) == (
        float(interval_values["workpiece_width_mm"].value)
        + 2.0 * float(interval_values["side_clearance_mm"].value)
        + 1.0
    )

    conditional_values = representative["conditional_correlation"].ground_truth
    fragile = bool(conditional_values["workpiece_fragile"].value)
    assert float(conditional_values["positioning_accuracy_mm"].value) == (
        0.2 if fragile else 2.0
    )
    assert float(conditional_values["repeatability_mm"].value) == (
        0.1 if fragile else 1.0
    )
