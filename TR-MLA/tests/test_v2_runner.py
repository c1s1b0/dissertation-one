from pathlib import Path

import numpy as np
import pytest

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.v2_runner import (
    run_v2_policies,
    run_v2_selected_experiment,
    run_v2_validation_tuning,
    select_v2_lambdas,
    summarize_v2_results,
)
from paper1_active_req.generator.complete_specs import generate_complete_spec
from paper1_active_req.generator.perturbations import mask_spec
from paper1_active_req.generator.v2_pivotal_specs import (
    V2AuditMetadata,
    V2RequirementTask,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _write_jsonl(path: Path, models: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(model.model_dump_json() for model in models) + "\n",
        encoding="utf-8",
    )


def test_v2_runner_rejects_v1_artifact_paths(tmp_path: Path):
    """防止 v2 runner 因路径误配而读取冻结的 v1 结果或数据。"""
    with pytest.raises(ValueError, match="v2 paths"):
        run_v2_policies(
            tasks_path=Path("data/generated/formal/test_tasks.jsonl"),
            training_specs_path=tmp_path / "v2" / "complete_specs.jsonl",
            schema_path=Path("configs/fields.yaml"),
            output_path=tmp_path / "v2" / "episodes.jsonl",
            seed=1,
            budget=1,
            lambda_cost=0.0,
            lambda_solver=0.0,
        )


def test_v2_runner_writes_eight_paired_policy_records_from_train_only(
    tmp_path: Path,
):
    """防止漏跑任一基线/消融，或把非 train 规格用于条件模型。"""
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(19)
    train_spec = generate_complete_spec(rng, schema, engine, "v2-spec-train")
    train_spec.split = "train"
    test_spec = generate_complete_spec(rng, schema, engine, "v2-spec-test")
    test_spec.split = "test"
    base_task = mask_spec(test_spec, missing_rate=0.3, rng=rng)
    task = V2RequirementTask.model_validate(
        {
            **base_task.model_dump(),
            "audit": V2AuditMetadata(
                module_id="load_space",
                pivot_field_id=None,
                affected_critical_field_ids=(),
                mechanism="interval_reduction",
            ).model_dump(),
        }
    )
    data_dir = tmp_path / "v2" / "data"
    task_path = data_dir / "test_tasks.jsonl"
    specs_path = data_dir / "complete_specs.jsonl"
    output_path = tmp_path / "v2" / "results" / "episodes.jsonl"
    _write_jsonl(task_path, [task])
    _write_jsonl(specs_path, [train_spec, test_spec])

    written = run_v2_policies(
        tasks_path=task_path,
        training_specs_path=specs_path,
        schema_path=Path("configs/fields.yaml"),
        output_path=output_path,
        seed=19,
        budget=1,
        lambda_cost=0.0,
        lambda_solver=0.0,
    )

    names = {
        __import__("json").loads(line)["policy_name"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    }
    assert written == 8
    assert names == {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "ccla_one_step",
        "ccla_two_step",
        "ccla_two_step_unconditioned",
    }


def test_v2_runner_can_limit_validation_candidate_to_ccla_two_step(
    tmp_path: Path,
):
    """防止 λ 调参重复运行与候选无关的七个基线。"""
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(23)
    train_spec = generate_complete_spec(rng, schema, engine, "v2-spec-train")
    validation_spec = generate_complete_spec(rng, schema, engine, "v2-spec-validation")
    train_spec.split = "train"
    validation_spec.split = "validation"
    task = V2RequirementTask.model_validate(
        {
            **mask_spec(validation_spec, missing_rate=0.3, rng=rng).model_dump(),
            "audit": V2AuditMetadata(
                module_id="load_space", pivot_field_id=None,
                affected_critical_field_ids=(), mechanism="interval_reduction",
            ).model_dump(),
        }
    )
    data_dir = tmp_path / "v2" / "data"
    _write_jsonl(data_dir / "validation_tasks.jsonl", [task])
    _write_jsonl(data_dir / "complete_specs.jsonl", [train_spec, validation_spec])
    output_path = tmp_path / "v2" / "tuning" / "candidate.jsonl"

    written = run_v2_policies(
        tasks_path=data_dir / "validation_tasks.jsonl",
        training_specs_path=data_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        output_path=output_path,
        seed=23,
        budget=1,
        lambda_cost=0.0,
        lambda_solver=0.0,
        policy_names={"ccla_two_step"},
    )

    assert written == 1
    assert {
        __import__("json").loads(line)["policy_name"]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    } == {"ccla_two_step"}


def test_v2_summary_writes_overall_and_missing_rate_paired_comparisons(
    tmp_path: Path,
):
    """防止汇总遗漏 CCLA-2 对两个预注册主比较对象的配对统计。"""
    records_path = tmp_path / "v2" / "episodes.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task_id, missing_rate, values in [
        ("task-a", 0.3, {"ccla_two_step": 1.0, "criticality_greedy": 0.5, "ccla_one_step": 0.6}),
        ("task-b", 0.7, {"ccla_two_step": 0.8, "criticality_greedy": 0.4, "ccla_one_step": 0.5}),
    ]:
        for policy_name, crr in values.items():
            rows.append(
                {
                    "task_id": task_id,
                    "missing_rate": missing_rate,
                    "policy_name": policy_name,
                    "metrics": {"crr_at_8": crr, "answer_cost": 1.0},
                }
            )
    records_path.write_text(
        "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "v2" / "summary"

    summarize_v2_results(records_path, output_dir)

    comparisons = __import__("pandas").read_csv(output_dir / "pairwise_tests.csv")
    assert set(comparisons["baseline"]) == {"criticality_greedy", "ccla_one_step"}
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "stratified_pairwise_tests.csv").exists()


def test_v2_lambda_selection_uses_validation_crr_then_auc_then_cost():
    """防止选择器用 test 信息或在主指标并列时随机挑选候选。"""
    selected = select_v2_lambdas(
        [
            {"lambda_cost": 0.0, "lambda_solver": 0.0, "crr_at_8": 0.8, "recall_budget_auc": 0.7, "answer_cost": 2.0},
            {"lambda_cost": 0.1, "lambda_solver": 0.0, "crr_at_8": 0.8, "recall_budget_auc": 0.8, "answer_cost": 3.0},
            {"lambda_cost": 0.2, "lambda_solver": 0.0, "crr_at_8": 0.8, "recall_budget_auc": 0.8, "answer_cost": 1.0},
        ]
    )

    assert selected["selection_split"] == "validation"
    assert selected["selected_lambdas"] == {
        "lambda_cost": 0.2,
        "lambda_solver": 0.0,
    }


def test_v2_validation_tuning_rejects_test_tasks(tmp_path: Path):
    """防止调参函数将 test 任务作为候选 λ 的选择依据。"""
    task_path = tmp_path / "v2" / "data" / "test_tasks.jsonl"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        '{"split":"test"}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="validation"):
        run_v2_validation_tuning(
            validation_tasks_path=task_path,
            training_specs_path=tmp_path / "v2" / "data" / "complete_specs.jsonl",
            schema_path=Path("configs/fields.yaml"),
            output_dir=tmp_path / "v2" / "tuning",
            seed=20260805,
            budget=8,
        )


def test_v2_selected_experiment_rejects_non_validation_selection_file(
    tmp_path: Path,
):
    """防止正式运行加载来源不明或由 test 选择得到的 λ 文件。"""
    selected_path = tmp_path / "v2" / "tuning" / "selected_lambdas.yaml"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        "selection_split: test\nselected_lambdas:\n  lambda_cost: 0.0\n  lambda_solver: 0.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validation"):
        run_v2_selected_experiment(
            tasks_path=tmp_path / "v2" / "data" / "test_tasks.jsonl",
            training_specs_path=tmp_path / "v2" / "data" / "complete_specs.jsonl",
            schema_path=Path("configs/fields.yaml"),
            selected_lambdas_path=selected_path,
            output_path=tmp_path / "v2" / "formal" / "episodes.jsonl",
            seed=20260805,
            budget=8,
        )
