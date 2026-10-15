from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldKind,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.experiment.v3_episode import V3EpisodeResult, V3EpisodeStep
from paper1_active_req.generator.v3_relational_specs import (
    V3AuditMetadata,
    V3RequirementTask,
)
from paper1_active_req.solver.engine import SolverResult


@dataclass
class StaticInferenceModel:
    inferred: dict[str, object]

    def reliably_resolved(self, state: RequirementState) -> dict[str, object]:
        del state
        return dict(self.inferred)


class SatisfiableEngine:
    def check(self, state: RequirementState) -> SolverResult:
        del state
        return SolverResult(satisfiable=True)


def _value(value: object | None, status: FieldStatus) -> FieldValue:
    return FieldValue(value=value, status=status, source="test")


def _task_and_schema() -> tuple[V3RequirementTask, dict[str, FieldDefinition]]:
    schema = {
        "critical": FieldDefinition(
            field_id="critical",
            kind=FieldKind.CATEGORICAL,
            categories=("yes", "no"),
            criticality=1.0,
            answer_cost=2.0,
        ),
        "pivot": FieldDefinition(
            field_id="pivot",
            kind=FieldKind.CATEGORICAL,
            categories=("left", "right"),
            criticality=1.0,
            answer_cost=1.0,
        ),
        "noncritical": FieldDefinition(
            field_id="noncritical",
            kind=FieldKind.CATEGORICAL,
            categories=("a", "b"),
            criticality=0.0,
            answer_cost=3.0,
        ),
    }
    task = V3RequirementTask(
        task_id="task-1",
        spec_id="spec-1",
        split="test",
        missing_rate=0.7,
        initial_state=RequirementState(
            values={field_id: _value(None, FieldStatus.HIDDEN) for field_id in schema}
        ),
        ground_truth={
            "critical": _value("yes", FieldStatus.OBSERVED),
            "pivot": _value("left", FieldStatus.OBSERVED),
            "noncritical": _value("a", FieldStatus.OBSERVED),
        },
        audit=V3AuditMetadata(
            module_id="v3.test",
            pivot_field_id="pivot",
            affected_critical_field_ids=("critical",),
            mechanism="interface_compatibility",
        ),
    )
    return task, schema


def _episode(task: V3RequirementTask, *, selected: str = "critical") -> V3EpisodeResult:
    answer = task.ground_truth[selected]
    final_state = task.initial_state.apply_answer(selected, answer)
    return V3EpisodeResult(
        task_id=task.task_id,
        policy_name="tr_mla",
        initial_state=task.initial_state,
        final_state=final_state,
        steps=[
            V3EpisodeStep(
                step_index=0,
                field_id=selected,
                answer=answer,
                satisfiable_after_answer=True,
            )
        ],
        solver_calls=2,
    )


def test_v3_metrics_replays_all_budgets_and_credits_direct_correct_answers():
    """删除重放或直接答案计分时，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    metrics = v3_episode_metrics(
        _episode(task), task, schema, StaticInferenceModel({}), SatisfiableEngine()
    )

    assert metrics["crr_at_0"] == 0.0
    assert metrics["crr_at_8"] == 0.5
    assert metrics["crr_auc"] == pytest.approx(0.46875)
    assert metrics["answer_cost"] == 2.0
    assert metrics["solver_calls"] == 2.0
    assert metrics["hard_constraint_satisfied"] == 1.0


def test_wrong_confident_inference_scores_zero_crr_and_precision():
    """若错误但高置信的推断被记为正确，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    result = V3EpisodeResult(
        task_id=task.task_id,
        policy_name="tr_mla",
        initial_state=task.initial_state,
        final_state=task.initial_state,
        steps=[],
        solver_calls=1,
    )
    metrics = v3_episode_metrics(
        result,
        task,
        schema,
        StaticInferenceModel({"critical": "no", "pivot": "right"}),
        SatisfiableEngine(),
    )

    assert metrics["crr_at_8"] == 0.0
    assert metrics["inference_precision"] == 0.0
    assert metrics["inference_coverage"] == pytest.approx(2 / 3)


def test_v3_metrics_records_noncritical_and_pivot_queries():
    """若非关键询问或枢纽选择没有记录，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    metrics = v3_episode_metrics(
        _episode(task, selected="pivot"),
        task,
        schema,
        StaticInferenceModel({}),
        SatisfiableEngine(),
    )

    assert metrics["pivot_selected"] == 1.0
    assert metrics["noncritical_query_count"] == 0.0
    assert metrics["runtime_seconds"] == 0.0


def test_correct_reliable_inference_receives_crr_and_precision_credit():
    """若可靠且正确的推断不计入 CRR/precision，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    result = V3EpisodeResult(
        task_id=task.task_id,
        policy_name="tr_mla",
        initial_state=task.initial_state,
        final_state=task.initial_state,
        steps=[],
        solver_calls=1,
    )

    metrics = v3_episode_metrics(
        result,
        task,
        schema,
        StaticInferenceModel({"critical": "yes"}),
        SatisfiableEngine(),
    )

    assert metrics["crr_at_8"] == 0.5
    assert metrics["inference_precision"] == 1.0
    assert metrics["inference_coverage"] == pytest.approx(1 / 3)


def test_wrong_direct_answer_does_not_receive_crr_credit():
    """若错误的直接回答被当作观察真值，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    wrong_answer = _value("no", FieldStatus.OBSERVED)
    result = V3EpisodeResult(
        task_id=task.task_id,
        policy_name="tr_mla",
        initial_state=task.initial_state,
        final_state=task.initial_state.apply_answer("critical", wrong_answer),
        steps=[
            V3EpisodeStep(
                step_index=0,
                field_id="critical",
                answer=wrong_answer,
                satisfiable_after_answer=True,
            )
        ],
        solver_calls=2,
    )

    metrics = v3_episode_metrics(
        result, task, schema, StaticInferenceModel({}), SatisfiableEngine()
    )

    assert metrics["crr_at_8"] == 0.0


def test_v3_metrics_ignores_steps_after_budget_eight():
    """若第九步被纳入 CRR@8 或成本，本测试应失败。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    wrong_answer = _value("no", FieldStatus.OBSERVED)
    correct_answer = task.ground_truth["critical"]
    steps = [
        V3EpisodeStep(
            step_index=index,
            field_id="critical",
            answer=wrong_answer if index < 8 else correct_answer,
            satisfiable_after_answer=True,
        )
        for index in range(9)
    ]
    result = V3EpisodeResult(
        task_id=task.task_id,
        policy_name="tr_mla",
        initial_state=task.initial_state,
        final_state=task.initial_state,
        steps=steps,
        solver_calls=18,
    )

    metrics = v3_episode_metrics(
        result, task, schema, StaticInferenceModel({}), SatisfiableEngine()
    )

    assert metrics["crr_at_8"] == 0.0
    assert metrics["answer_cost"] == 16.0


def test_frozen_critical_fields_use_real_schema_threshold_status_and_queryability():
    """K 必须只保留初始未解析、可询问且 criticality >= 0.8 的真实字段。"""
    from paper1_active_req.domain.schema import load_field_schema
    from paper1_active_req.evaluation.v3_protocol import frozen_critical_field_ids

    schema = load_field_schema(Path("configs/fields.yaml"))
    state = RequirementState(
        values={
            "workpiece_mass_kg": _value(None, FieldStatus.HIDDEN),
            "workpiece_length_mm": _value(None, FieldStatus.CONFLICTING),
            "workpiece_height_mm": _value(None, FieldStatus.INFERRED),
            "available_height_mm": _value(None, FieldStatus.HIDDEN),
            "side_clearance_mm": _value(None, FieldStatus.HIDDEN),
            "workpiece_fragile": _value(None, FieldStatus.HIDDEN),
            "power_supply": _value(None, FieldStatus.HIDDEN),
            "available_width_mm": _value(300.0, FieldStatus.OBSERVED),
        }
    )

    assert frozen_critical_field_ids(state, schema) == (
        "workpiece_mass_kg",
        "workpiece_length_mm",
        "workpiece_height_mm",
    )


def test_low_criticality_query_is_counted_but_has_no_crr_credit():
    """0.5/0.6/0.7 字段可以被问到，但不能进入 K 或获得 CRR。"""
    from paper1_active_req.evaluation.v3_metrics import v3_episode_metrics

    task, schema = _task_and_schema()
    schema["noncritical"] = schema["noncritical"].model_copy(update={"criticality": 0.7})
    metrics = v3_episode_metrics(
        _episode(task, selected="noncritical"),
        task,
        schema,
        StaticInferenceModel({}),
        SatisfiableEngine(),
    )

    assert metrics["crr_at_8"] == 0.0
    assert metrics["noncritical_query_count"] == 1.0
