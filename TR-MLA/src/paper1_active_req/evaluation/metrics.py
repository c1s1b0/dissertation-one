from __future__ import annotations

import math

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldStatus,
)
from paper1_active_req.experiment.episode import EpisodeResult
from paper1_active_req.generator.complete_specs import RequirementTask
from paper1_active_req.solver.engine import ConstraintEngine


CRITICALITY_THRESHOLD = 0.8


def critical_requirement_recall(
    result: EpisodeResult,
    task: RequirementTask,
    schema: dict[str, FieldDefinition],
    *,
    budget: int,
) -> float:
    unresolved = {
        field_id
        for field_id, value in task.initial_state.values.items()
        if value.status
        in {
            FieldStatus.HIDDEN,
            FieldStatus.CONFLICTING,
            FieldStatus.INFERRED,
        }
        and schema[field_id].queryable
        and schema[field_id].criticality >= CRITICALITY_THRESHOLD
    }
    denominator = sum(
        schema[field_id].criticality for field_id in unresolved
    )
    if denominator == 0.0:
        return 1.0

    recovered: set[str] = set()
    for step in result.steps[:budget]:
        field_id = step.field_id
        if field_id not in unresolved:
            continue
        truth = task.ground_truth[field_id].value
        if _values_equal(step.answer.value, truth):
            recovered.add(field_id)
    numerator = sum(
        schema[field_id].criticality for field_id in recovered
    )
    return numerator / denominator


def recall_budget_auc(
    result: EpisodeResult,
    task: RequirementTask,
    schema: dict[str, FieldDefinition],
    *,
    max_budget: int = 8,
) -> float:
    if max_budget <= 0:
        raise ValueError("max_budget must be positive")
    recalls = [
        critical_requirement_recall(
            result, task, schema, budget=budget
        )
        for budget in range(max_budget + 1)
    ]
    area = sum(
        (left + right) / 2.0
        for left, right in zip(recalls, recalls[1:])
    )
    return area / float(max_budget)


def episode_metrics(
    result: EpisodeResult,
    task: RequirementTask,
    schema: dict[str, FieldDefinition],
    engine: ConstraintEngine,
) -> dict[str, float]:
    initial_solver = engine.check(task.initial_state)
    final_solver = engine.check(result.final_state)
    inferred = [
        (field_id, value)
        for field_id, value in result.final_state.values.items()
        if value.status == FieldStatus.INFERRED
    ]
    unsupported = sum(
        not _values_equal(
            value.value, task.ground_truth[field_id].value
        )
        for field_id, value in inferred
    )
    unsupported_rate = unsupported / len(inferred) if inferred else 0.0
    return {
        "crr_at_8": critical_requirement_recall(
            result, task, schema, budget=8
        ),
        "recall_budget_auc": recall_budget_auc(
            result, task, schema, max_budget=8
        ),
        "hard_constraint_satisfied": float(
            final_solver.satisfiable is True
        ),
        "conflict_detected": float(
            task.injected_conflict_type is not None
            and initial_solver.satisfiable is False
        ),
        "unsupported_assumption_rate": float(unsupported_rate),
        "answer_cost": float(
            sum(
                schema[step.field_id].answer_cost
                for step in result.steps
            )
        ),
        "solver_calls": float(result.solver_calls),
        "questions_asked": float(len(result.steps)),
    }


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(
        right, (int, float)
    ):
        return math.isclose(
            float(left), float(right), rel_tol=1e-9, abs_tol=1e-9
        )
    return left == right
