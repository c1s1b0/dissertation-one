from __future__ import annotations

import math

import numpy as np

from paper1_active_req.domain.models import (
    FieldKind,
    RequirementState,
)
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.solver.engine import ConstraintEngine, SolverResult

from .base import EmpiricalAnswerModel, PolicyDecision


def _candidate_ids(
    state: RequirementState, context: PolicyContext
) -> list[str]:
    return [
        field_id
        for field_id in state.queryable_field_ids()
        if context.field_schema[field_id].queryable
    ]


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        candidates = _candidate_ids(state, context)
        if not candidates:
            return PolicyDecision(field_id=None)
        selected = str(self._rng.choice(candidates))
        return PolicyDecision(
            field_id=selected,
            utility_terms={"random_rank": 1.0},
        )


class FixedQuestionnairePolicy:
    name = "fixed_questionnaire"

    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order or [
            "workpiece_mass_kg",
            "workpiece_length_mm",
            "workpiece_width_mm",
            "target_throughput_per_hour",
            "travel_distance_mm",
            "positioning_accuracy_mm",
            "available_width_mm",
            "upstream_output_interface",
            "downstream_input_interface",
        ]

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        candidates = set(_candidate_ids(state, context))
        selected = next(
            (field_id for field_id in self.order if field_id in candidates),
            min(candidates) if candidates else None,
        )
        return PolicyDecision(
            field_id=selected,
            utility_terms={"fixed_rank": 1.0} if selected else {},
        )


class CriticalityGreedyPolicy:
    name = "criticality_greedy"

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        candidates = _candidate_ids(state, context)
        if not candidates:
            return PolicyDecision(field_id=None)
        selected = min(
            candidates,
            key=lambda field_id: (
                -context.field_schema[field_id].criticality,
                field_id,
            ),
        )
        return PolicyDecision(
            field_id=selected,
            utility_terms={
                "criticality": context.field_schema[selected].criticality
            },
        )


class UncertaintyAFAPolicy:
    name = "uncertainty_afa"

    def __init__(self, empirical_model: EmpiricalAnswerModel) -> None:
        self.empirical_model = empirical_model

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        scores: dict[str, float] = {}
        for field_id in _candidate_ids(state, context):
            answers = self.empirical_model.answers_for(field_id)
            if not answers:
                continue
            total = sum(answer.probability for answer in answers)
            probabilities = [
                answer.probability / total for answer in answers
            ]
            scores[field_id] = -sum(
                probability * math.log(probability)
                for probability in probabilities
            )
        if not scores:
            return PolicyDecision(field_id=None)
        selected = min(scores, key=lambda item: (-scores[item], item))
        return PolicyDecision(
            field_id=selected,
            utility_terms={"entropy": scores[selected]},
        )


class ConstraintRangeOnlyPolicy:
    name = "constraint_range_only"

    def __init__(
        self,
        engine: ConstraintEngine,
        empirical_model: EmpiricalAnswerModel,
    ) -> None:
        self.engine = engine
        self.empirical_model = empirical_model

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        current_width = _state_width(solver_result, context)
        scores: dict[str, float] = {}
        for field_id in _candidate_ids(state, context):
            answers = self.empirical_model.answers_for(field_id)
            if not answers:
                continue
            total = sum(answer.probability for answer in answers)
            expected_width = 0.0
            for answer in answers:
                probability = answer.probability / total
                counterfactual = self.engine.counterfactual(
                    state, field_id, answer.value
                )
                next_width = (
                    _state_width(counterfactual, context)
                    if counterfactual.satisfiable is True
                    else current_width
                )
                expected_width += probability * next_width
            scores[field_id] = current_width - expected_width
        if not scores:
            return PolicyDecision(field_id=None)
        selected = min(scores, key=lambda item: (-scores[item], item))
        return PolicyDecision(
            field_id=selected,
            utility_terms={"domain_reduction": scores[selected]},
        )


def _state_width(
    solver_result: SolverResult, context: PolicyContext
) -> float:
    total = 0.0
    for field_id, definition in context.field_schema.items():
        if definition.kind not in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
            continue
        if field_id not in solver_result.domains:
            continue
        lower, upper = solver_result.domains[field_id]
        base_width = float(definition.upper) - float(definition.lower)
        ratio = max(float(upper) - float(lower), 1e-12) / base_width
        total += definition.criticality * math.log(ratio + 1e-12)
    return total
