from __future__ import annotations

import math

from paper1_active_req.domain.models import FieldDefinition, FieldStatus
from paper1_active_req.experiment.episode import EpisodeResult
from paper1_active_req.generator.v2_pivotal_specs import V2RequirementTask
from paper1_active_req.policies.lookahead import ConditionalAnswerModel
from paper1_active_req.solver.engine import ConstraintEngine


CRITICALITY_THRESHOLD = 0.8


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _state_at_budget(result: EpisodeResult, budget: int):
    state = result.initial_state.public_copy()
    for step in result.steps[:budget]:
        state = state.apply_answer(step.field_id, step.answer)
    return state


def v2_critical_requirement_recall(
    result: EpisodeResult,
    task: V2RequirementTask,
    schema: dict[str, FieldDefinition],
    answer_model: ConditionalAnswerModel,
    *,
    budget: int,
) -> float:
    """按预算评估直接回答或正确唯一条件推断的关键字段恢复率。"""
    unresolved = {
        field_id
        for field_id, value in task.initial_state.values.items()
        if value.status in {FieldStatus.HIDDEN, FieldStatus.CONFLICTING, FieldStatus.INFERRED}
        and schema[field_id].queryable
        and schema[field_id].criticality >= CRITICALITY_THRESHOLD
    }
    denominator = sum(schema[field_id].criticality for field_id in unresolved)
    if denominator == 0.0:
        return 1.0
    state = _state_at_budget(result, budget)
    recovered = {
        field_id
        for field_id in unresolved
        if state.values[field_id].status == FieldStatus.OBSERVED
        and _values_equal(state.values[field_id].value, task.ground_truth[field_id].value)
    }
    for field_id, inferred in answer_model.uniquely_determined_values(state).items():
        if field_id in unresolved and _values_equal(inferred, task.ground_truth[field_id].value):
            recovered.add(field_id)
    return sum(schema[field_id].criticality for field_id in recovered) / denominator


def v2_recall_budget_auc(
    result: EpisodeResult,
    task: V2RequirementTask,
    schema: dict[str, FieldDefinition],
    answer_model: ConditionalAnswerModel,
    *,
    max_budget: int = 8,
) -> float:
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    recalls = [
        v2_critical_requirement_recall(result, task, schema, answer_model, budget=budget)
        for budget in range(max_budget + 1)
    ]
    return sum((left + right) / 2.0 for left, right in zip(recalls, recalls[1:])) / max_budget


def v2_episode_metrics(
    result: EpisodeResult,
    task: V2RequirementTask,
    schema: dict[str, FieldDefinition],
    answer_model: ConditionalAnswerModel,
    engine: ConstraintEngine,
) -> dict[str, float]:
    final_state = _state_at_budget(result, 8)
    determined = answer_model.uniquely_determined_values(final_state)
    critical_hidden = {
        field_id
        for field_id, value in task.initial_state.values.items()
        if value.status in {FieldStatus.HIDDEN, FieldStatus.CONFLICTING, FieldStatus.INFERRED}
        and schema[field_id].criticality >= CRITICALITY_THRESHOLD
    }
    determinacy = len(critical_hidden & set(determined)) / len(critical_hidden) if critical_hidden else 1.0
    pivot = task.audit.pivot_field_id
    pivot_selected = float(pivot is not None and any(step.field_id == pivot for step in result.steps[:8]))
    return {
        "crr_at_8": v2_critical_requirement_recall(result, task, schema, answer_model, budget=8),
        "recall_budget_auc": v2_recall_budget_auc(result, task, schema, answer_model),
        "critical_determinacy_rate": float(determinacy),
        "answer_cost": float(sum(schema[step.field_id].answer_cost for step in result.steps[:8])),
        "solver_calls": float(result.solver_calls),
        "questions_asked": float(len(result.steps[:8])),
        "pivot_selection_rate": pivot_selected,
        "hard_constraint_satisfied": float(engine.check(final_state).satisfiable is True),
    }
