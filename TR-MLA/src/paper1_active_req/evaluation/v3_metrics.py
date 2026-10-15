from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from typing import Protocol

import numpy as np

from paper1_active_req.domain.models import FieldDefinition, FieldKind, FieldStatus, FieldValue, RequirementState
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.evaluation.v3_protocol import (
    FROZEN_CRITICALITY_THRESHOLD,
    frozen_critical_field_ids,
)
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.solver.engine import ConstraintEngine


class ReliableInferenceModel(Protocol):
    """v3 指标所需的最小关系模型公开接口。"""

    def reliably_resolved(self, state: RequirementState) -> dict[str, object]: ...


def _matches_truth(
    predicted: object,
    truth: FieldValue,
    definition: FieldDefinition,
) -> bool:
    if truth.value is None:
        return predicted is None
    if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
        if not isinstance(predicted, (int, float)) or not isinstance(truth.value, (int, float)):
            return False
        return isclose(float(predicted), float(truth.value), rel_tol=0.0, abs_tol=1e-9)
    return predicted == truth.value


def _replay_state(result: V3EpisodeResult, budget: int) -> RequirementState:
    state = result.initial_state.public_copy()
    for step in result.steps[:budget]:
        state = state.apply_answer(step.field_id, step.answer)
    return state


def _crr(
    state: RequirementState,
    task: V3RequirementTask,
    schema: Mapping[str, FieldDefinition],
    model: ReliableInferenceModel,
) -> tuple[float, int, int]:
    """返回 CRR、可靠推断总数与其中正确数。"""
    frozen_ids = frozen_critical_field_ids(task.initial_state, schema)
    critical_total = sum(schema[field_id].criticality for field_id in frozen_ids)
    if critical_total <= 0.0:
        raise ValueError("CRR requires positive total criticality")

    inferred = model.reliably_resolved(state.public_copy())
    correct_weight = 0.0
    inferred_total = 0
    inferred_correct = 0
    for field_id in frozen_ids:
        definition = schema[field_id]
        truth = task.ground_truth.get(field_id)
        state_value = state.values.get(field_id)
        if truth is None or state_value is None:
            raise ValueError(f"task/state missing schema field: {field_id}")
        direct_correct = (
            state_value.status == FieldStatus.OBSERVED
            and _matches_truth(state_value.value, truth, definition)
        )
        inferred_value = inferred.get(field_id)
        inferred_correct_for_field = (
            not direct_correct
            and field_id in inferred
            and _matches_truth(inferred_value, truth, definition)
        )
        if not direct_correct and field_id in inferred:
            inferred_total += 1
            inferred_correct += int(inferred_correct_for_field)
        if direct_correct or inferred_correct_for_field:
            correct_weight += definition.criticality
    return correct_weight / critical_total, inferred_total, inferred_correct


def v3_episode_metrics(
    result: V3EpisodeResult,
    task: V3RequirementTask,
    schema: Mapping[str, FieldDefinition],
    model: ReliableInferenceModel,
    engine: ConstraintEngine,
) -> dict[str, float]:
    """从一个 v3 episode 重放 0--8 轮并生成确认性及辅助指标。

    直接观察只在答案与 task 真值一致时计分；关系推断也必须通过
    ``reliably_resolved`` 且与真值一致。该函数不读取数据文件或任何正式工件。
    """
    if result.task_id != task.task_id:
        raise ValueError("episode result and task_id must match")
    if set(schema) != set(task.ground_truth):
        raise ValueError("schema and task ground truth fields must match")

    crr_by_budget: list[float] = []
    final_inferred_total = 0
    final_inferred_correct = 0
    for budget in range(9):
        crr, inferred_total, inferred_correct = _crr(
            _replay_state(result, budget), task, schema, model
        )
        crr_by_budget.append(crr)
        if budget == 8:
            final_inferred_total = inferred_total
            final_inferred_correct = inferred_correct

    answered_steps = result.steps[:8]
    hidden_initially = sum(
        value.status != FieldStatus.OBSERVED
        for value in result.initial_state.values.values()
    )
    final_solver = engine.check(_replay_state(result, 8))
    pivot_field_id = task.audit.pivot_field_id
    runtime = getattr(result, "runtime_seconds", 0.0)
    if not isinstance(runtime, (int, float)):
        raise ValueError("runtime_seconds must be numeric when present")

    metrics = {
        f"crr_at_{budget}": float(value)
        for budget, value in enumerate(crr_by_budget)
    }
    metrics.update(
        {
            "crr_auc": float(np.trapz(crr_by_budget, dx=1.0) / 8.0),
            "answer_cost": float(
                sum(schema[step.field_id].answer_cost for step in answered_steps)
            ),
            "noncritical_query_count": float(
                sum(
                    schema[step.field_id].criticality < FROZEN_CRITICALITY_THRESHOLD
                    for step in answered_steps
                )
            ),
            "inference_precision": (
                float(final_inferred_correct / final_inferred_total)
                if final_inferred_total
                else 0.0
            ),
            "inference_coverage": (
                float(final_inferred_total / hidden_initially)
                if hidden_initially
                else 0.0
            ),
            "pivot_selected": float(
                pivot_field_id is not None
                and any(step.field_id == pivot_field_id for step in answered_steps)
            ),
            "runtime_seconds": float(runtime),
            "solver_calls": float(result.solver_calls),
            "hard_constraint_satisfied": float(final_solver.satisfiable is True),
        }
    )
    return metrics
