from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import z3

import paper1_active_req.experiment.v3_runner as v3_runner
from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.v3_runner import (
    V3ExperimentRecord,
    run_v3_policies,
    run_v3_validation_tuning,
    select_v3_lambda_candidate,
    summarize_v3_results,
    validate_v3_paths,
)
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.generator.complete_specs import (
    CompleteSpec,
    generate_complete_spec,
)
from paper1_active_req.generator.v3_relational_specs import (
    V3AuditMetadata,
    V3RequirementTask,
)
from paper1_active_req.solver.engine import ConstraintEngine, SolverResult


SCHEMA_PATH = Path("configs/fields.yaml")
RELATION_CONFIG_PATH = Path("configs/v3_relations.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "\n".join(
            value.model_dump_json() if hasattr(value, "model_dump_json") else json.dumps(value)
            for value in values
        )
        + "\n",
        encoding="utf-8",
    )


def _runner_fixture(tmp_path: Path, *, rates: tuple[float, ...] = (0.5,)) -> tuple[Path, Path, Path]:
    """构造真实 v3 任务；若实现开始读取非训练/真值泄露，此 fixture 会失效。"""
    schema = load_field_schema(SCHEMA_PATH)
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(19)
    training = [
        generate_complete_spec(rng, schema, engine, f"train-{index}").model_copy(
            update={"split": "train"}
        )
        for index in range(3)
    ]
    truth = training[0].values
    state_values = {key: value.model_copy(deep=True) for key, value in truth.items()}
    state_values["workpiece_mass_kg"] = FieldValue(
        value=None, status=FieldStatus.HIDDEN, source="runner-test"
    )
    tasks = [
        V3RequirementTask(
            task_id=f"validation-1-m{int(rate * 100)}",
            spec_id="validation-1",
            split="validation",
            missing_rate=rate,
            initial_state=RequirementState(values=state_values),
            ground_truth={key: value.model_copy(deep=True) for key, value in truth.items()},
            audit=V3AuditMetadata(
                module_id="v3.interface_compatibility",
                pivot_field_id=None,
                mechanism="interface_compatibility",
            ),
        )
        for rate in rates
    ]
    training_path = tmp_path / "training_specs.jsonl"
    tasks_path = tmp_path / "validation_tasks.jsonl"
    _write_jsonl(training_path, training)
    _write_jsonl(tasks_path, tasks)
    return tasks_path, training_path, tmp_path / "episodes.jsonl"


def test_v3_runner_rejects_v1_v2_and_formal_paths(tmp_path: Path):
    """删除 v3 路径隔离会允许旧实验或正式输出污染本轮记录。"""
    for bad in (
        Path("data/generated/v1/test_tasks.jsonl"),
        Path("data/generated/v2/test_tasks.jsonl"),
        Path("results/v3/formal/episodes.jsonl"),
    ):
        with pytest.raises(ValueError, match="v3 paths"):
            validate_v3_paths(bad, tmp_path / "episodes.jsonl")


def test_v3_runner_produces_ten_paired_policy_records_for_each_task(tmp_path: Path):
    """删除任一策略工厂分支会使同一任务不再有十条配对轨迹。"""
    tasks_path, training_path, output_path = _runner_fixture(tmp_path)

    written = run_v3_policies(
        tasks_path=tasks_path,
        training_specs_path=training_path,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        output_path=output_path,
        seed=20260806,
        budget=1,
        lambda_cost=0.05,
    )

    records = [
        V3ExperimentRecord.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == 10
    assert {record.policy_name for record in records} == {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "ccla_one_step",
        "v2_style_two_step_unconditioned",
        "marginal_lookahead_no_relations",
        "typed_relational_myopic",
        "tr_mla",
    }
    assert {record.task_id for record in records} == {"validation-1-m50"}
    assert all(record.split == "validation" for record in records)


def test_v3_runner_trmla_only_does_not_construct_unselected_policy_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """若仅调 TR-MLA 仍创建经验基线模型，validation-only 调参会无谓构造其余九策略依赖。"""
    tasks_path, training_path, output_path = _runner_fixture(tmp_path)

    def forbidden_empirical_model(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("TR-MLA-only runner constructed an unselected empirical model")

    monkeypatch.setattr(
        v3_runner.EmpiricalAnswerModel,
        "from_complete_specs",
        forbidden_empirical_model,
    )

    written = run_v3_policies(
        tasks_path=tasks_path,
        training_specs_path=training_path,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        output_path=output_path,
        seed=20260806,
        budget=1,
        lambda_cost=0.05,
        policy_names={"tr_mla"},
    )

    records = [
        V3ExperimentRecord.model_validate_json(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written == 1
    assert [record.policy_name for record in records] == ["tr_mla"]


def test_v3_runner_base_solver_preserves_real_trmla_trace_and_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """若预编译约束改变任一反事实可行性，真实 TR-MLA 的分支、得分或选问顺序会漂移。"""
    class LegacyConstraintEngine(ConstraintEngine):
        def check_satisfiability(self, state: RequirementState):
            cache_key = self._state_cache_key(state)
            cached = self._satisfiability_cache.get(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return cached.model_copy(deep=True)
            self._cache_misses += 1
            solver = z3.Solver()
            solver.set(timeout=self.timeout_ms)
            solver.add(*self._plain_constraints(state))
            status = solver.check()
            if status == z3.unknown:
                result = SolverResult(satisfiable=None, timed_out=True)
            else:
                result = SolverResult(satisfiable=status == z3.sat)
            self._satisfiability_cache[cache_key] = result.model_copy(deep=True)
            return result

    legacy_dir = tmp_path / "legacy"
    current_dir = tmp_path / "current"
    legacy_dir.mkdir()
    current_dir.mkdir()
    legacy_tasks, legacy_training, legacy_output = _runner_fixture(legacy_dir, rates=(0.5,))
    current_tasks, current_training, current_output = _runner_fixture(current_dir, rates=(0.5,))

    with monkeypatch.context() as scoped:
        scoped.setattr(v3_runner, "ConstraintEngine", LegacyConstraintEngine)
        run_v3_policies(
            tasks_path=legacy_tasks,
            training_specs_path=legacy_training,
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            output_path=legacy_output,
            seed=20260806,
            budget=2,
            lambda_cost=0.05,
            policy_names={"tr_mla"},
        )
    run_v3_policies(
        tasks_path=current_tasks,
        training_specs_path=current_training,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        output_path=current_output,
        seed=20260806,
        budget=2,
        lambda_cost=0.05,
        policy_names={"tr_mla"},
    )

    legacy_record = V3ExperimentRecord.model_validate_json(
        legacy_output.read_text(encoding="utf-8").strip()
    )
    current_record = V3ExperimentRecord.model_validate_json(
        current_output.read_text(encoding="utf-8").strip()
    )
    assert current_record.episode.model_dump(mode="json") == legacy_record.episode.model_dump(
        mode="json"
    )
    assert current_record.metrics == pytest.approx(legacy_record.metrics)


def test_v3_validation_selection_uses_m50_m70_spec_means_then_auc_cost_and_lambda():
    """若直接按任务计分或改变并列顺序，候选选择会与冻结协议不一致。"""
    selected = select_v3_lambda_candidate(
        [
            {
                "lambda_cost": 0.0,
                "records": [
                    {"spec_id": "a", "missing_rate": 0.5, "crr_at_8": 0.6, "crr_auc": 0.9, "answer_cost": 1.0},
                    {"spec_id": "a", "missing_rate": 0.7, "crr_at_8": 0.6, "crr_auc": 0.9, "answer_cost": 1.0},
                ],
            },
            {
                "lambda_cost": 0.2,
                "records": [
                    {"spec_id": "a", "missing_rate": 0.5, "crr_at_8": 0.8, "crr_auc": 0.4, "answer_cost": 3.0},
                    {"spec_id": "a", "missing_rate": 0.7, "crr_at_8": 0.6, "crr_auc": 0.6, "answer_cost": 2.0},
                ],
            },
            {
                "lambda_cost": 0.1,
                "records": [
                    {"spec_id": "a", "missing_rate": 0.5, "crr_at_8": 0.7, "crr_auc": 0.6, "answer_cost": 2.0},
                    {"spec_id": "a", "missing_rate": 0.7, "crr_at_8": 0.7, "crr_auc": 0.4, "answer_cost": 3.0},
                ],
            },
            {
                "lambda_cost": 0.05,
                "records": [
                    {"spec_id": "a", "missing_rate": 0.5, "crr_at_8": 0.7, "crr_auc": 0.5, "answer_cost": 2.5},
                    {"spec_id": "a", "missing_rate": 0.7, "crr_at_8": 0.7, "crr_auc": 0.5, "answer_cost": 2.5},
                ],
            },
        ]
    )

    assert selected["lambda_cost"] == 0.05


def test_v3_validation_tuning_rejects_nonfrozen_seed_budget_and_short_roster(tmp_path: Path):
    """调参不能用测试缩小预算、任意随机种子或伪造的短 validation roster。"""
    tasks_path, training_path, _ = _runner_fixture(tmp_path, rates=(0.5, 0.7))
    output_dir = tmp_path / "tuning"

    for rejected_budget in (1, 2):
        with pytest.raises(ValueError, match="seed 20260806 and budget 8"):
            run_v3_validation_tuning(
                validation_tasks_path=tasks_path, training_specs_path=training_path,
                schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
                output_dir=output_dir, seed=20260806, budget=rejected_budget,
            )
    with pytest.raises(ValueError, match="60-spec/180-task"):
        run_v3_validation_tuning(
            validation_tasks_path=tasks_path, training_specs_path=training_path,
            schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
            output_dir=output_dir, seed=20260806, budget=8,
        )


def test_v3_summary_rejects_non_1800_formal_records_before_writing_tables(tmp_path: Path):
    """若删除 1800 条门槛，任何不完整的正式运行都会被汇总为结果表。"""
    episodes_path = tmp_path / "formal_records.jsonl"
    episodes_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="1800"):
        summarize_v3_results(
            episodes_path=episodes_path,
            output_dir=tmp_path / "summary",
        )
    assert not (tmp_path / "summary").exists()


def test_v3_formal_summary_writes_confirmatory_statistics_all_metrics_and_success_gate(
    tmp_path: Path,
):
    """1,800 条正向合成记录必须生成已接线的确认性统计，而不是仅均值表。"""
    episodes_path = tmp_path / "formal_records.jsonl"
    records: list[V3ExperimentRecord] = []
    for spec_number in range(60):
        mechanism = ("interface", "capacity", "timing")[spec_number % 3]
        for missing_rate in (0.3, 0.5, 0.7):
            for policy_name in v3_runner.V3_POLICY_NAMES:
                is_trmla = policy_name == "tr_mla"
                metrics = {
                    **{f"crr_at_{budget}": 0.6 + (0.1 if is_trmla else 0.0) for budget in range(9)},
                    "crr_auc": 0.6 + (0.1 if is_trmla else 0.0),
                    "answer_cost": 4.0,
                    "noncritical_query_count": 1.0,
                    "inference_precision": 0.96,
                    "inference_coverage": 0.5,
                    "pivot_selected": 1.0,
                    "runtime_seconds": 0.01,
                    "solver_calls": 12.0,
                    "hard_constraint_satisfied": 1.0,
                }
                records.append(
                    V3ExperimentRecord(
                        task_id=f"test-{spec_number}-m{int(missing_rate * 100)}",
                        spec_id=f"test-{spec_number}",
                        split="test",
                        missing_rate=missing_rate,
                        mechanism=mechanism,
                        pivot=True,
                        policy_name=policy_name,
                        config_hash="a" * 64,
                        model_hash="b" * 64,
                        episode=V3EpisodeResult(
                            task_id=f"test-{spec_number}-m{int(missing_rate * 100)}",
                            policy_name=policy_name,
                            initial_state=RequirementState(values={}),
                            final_state=RequirementState(values={}),
                            steps=[],
                            solver_calls=0,
                        ),
                        metrics=metrics,
                    )
                )
    _write_jsonl(episodes_path, records)

    summarize_v3_results(episodes_path=episodes_path, output_dir=tmp_path / "summary")

    primary = json.loads((tmp_path / "summary" / "confirmatory_primary.json").read_text(encoding="utf-8"))
    gate = json.loads((tmp_path / "summary" / "success_gate.json").read_text(encoding="utf-8"))
    overall = (tmp_path / "summary" / "overall.csv").read_text(encoding="utf-8")
    mechanism = (tmp_path / "summary" / "exploratory_mechanism.csv").read_text(encoding="utf-8")
    ablation = (tmp_path / "summary" / "exploratory_ablation.csv").read_text(encoding="utf-8")

    assert primary["sample_unit"] == "spec (m50/m70 within-spec mean)"
    assert primary["comparison"]["n_specs"] == 60
    assert primary["comparison"]["bootstrap_replicates"] == 10_000
    assert primary["comparison"]["mean_difference"] == pytest.approx(0.1)
    assert primary["comparison"]["ci_low"] > 0.0
    assert primary["comparison"]["p_value"] < 0.05
    assert primary["comparison"]["rank_biserial"] == pytest.approx(1.0)
    assert gate == {
        "ci_lower_positive": True,
        "hard_constraints_all_satisfied": True,
        "inference_precision_at_least_0_95": True,
        "p_value_below_0_05": True,
        "success": True,
    }
    for metric_name in (
        "noncritical_query_count",
        "inference_precision",
        "inference_coverage",
        "runtime_seconds",
        "solver_calls",
    ):
        assert metric_name in overall
    assert "holm_p_value" in mechanism
    assert "typed_relational_myopic" in ablation
