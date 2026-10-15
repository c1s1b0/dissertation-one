from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.solver.engine import ConstraintEngine, SolverResult

from .base import PolicyDecision, WeightedAnswer
from .baselines import _candidate_ids


def _value_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _weighted_answers(values: list[Any]) -> list[WeightedAnswer]:
    counts = Counter(values)
    total = sum(counts.values())
    return [
        WeightedAnswer(value=value, probability=count / total)
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    ]


@dataclass(frozen=True)
class ConditionalAnswerModel:
    """仅由训练规格建立的、基于公开同模块字段的条件回答分布。"""

    marginal_outcomes: dict[str, list[WeightedAnswer]]
    conditional_outcomes: dict[tuple[str, str, str], list[WeightedAnswer]]
    module_by_field: dict[str, str]
    use_module_conditioning: bool = True

    @classmethod
    def fit(
        cls,
        *,
        training_specs: list[dict[str, FieldValue]],
        module_by_field: dict[str, str],
        use_module_conditioning: bool = True,
    ) -> "ConditionalAnswerModel":
        marginal_values: dict[str, list[Any]] = {}
        conditional_values: dict[tuple[str, str, str], list[Any]] = {}
        for spec in training_specs:
            for target_field_id, target in spec.items():
                if target.value is None:
                    continue
                marginal_values.setdefault(target_field_id, []).append(target.value)
                if not use_module_conditioning:
                    continue
                target_module = module_by_field.get(target_field_id)
                if target_module is None:
                    continue
                for condition_field_id, condition in spec.items():
                    if (
                        condition_field_id == target_field_id
                        or condition.value is None
                        or module_by_field.get(condition_field_id) != target_module
                    ):
                        continue
                    key = (
                        target_field_id,
                        condition_field_id,
                        _value_key(condition.value),
                    )
                    conditional_values.setdefault(key, []).append(target.value)
        return cls(
            marginal_outcomes={
                field_id: _weighted_answers(values)
                for field_id, values in marginal_values.items()
            },
            conditional_outcomes={
                key: _weighted_answers(values)
                for key, values in conditional_values.items()
            },
            module_by_field=dict(module_by_field),
            use_module_conditioning=use_module_conditioning,
        )

    def answers_for(
        self, state: RequirementState, field_id: str
    ) -> list[WeightedAnswer]:
        if self.use_module_conditioning:
            module_id = self.module_by_field.get(field_id)
            if module_id is not None:
                for observed_field_id in sorted(state.values):
                    observed = state.values[observed_field_id]
                    if (
                        observed.status != FieldStatus.OBSERVED
                        or observed.value is None
                        or self.module_by_field.get(observed_field_id) != module_id
                    ):
                        continue
                    conditioned = self.conditional_outcomes.get(
                        (field_id, observed_field_id, _value_key(observed.value))
                    )
                    if conditioned:
                        return conditioned
        return self.marginal_outcomes.get(field_id, [])

    def uniquely_determined_values(
        self, state: RequirementState
    ) -> dict[str, Any]:
        determined: dict[str, Any] = {}
        for field_id, value in state.values.items():
            if value.status not in {FieldStatus.HIDDEN, FieldStatus.INFERRED}:
                continue
            answers = self.answers_for(state, field_id)
            if len(answers) == 1 and answers[0].probability == 1.0:
                determined[field_id] = answers[0].value
        return determined


class _CCLABasePolicy:
    def __init__(
        self,
        *,
        engine: ConstraintEngine,
        answer_model: ConditionalAnswerModel,
        lambda_cost: float,
        lambda_solver: float,
    ) -> None:
        if lambda_cost < 0.0 or lambda_solver < 0.0:
            raise ValueError("CCLA penalties must be non-negative")
        self.engine = engine
        self.answer_model = answer_model
        self.lambda_cost = lambda_cost
        self.lambda_solver = lambda_solver

    @staticmethod
    def _direct_gain(
        state: RequirementState, field_id: str, context: PolicyContext
    ) -> float:
        value = state.values[field_id]
        if value.status not in {FieldStatus.HIDDEN, FieldStatus.INFERRED}:
            return 0.0
        definition = context.field_schema[field_id]
        return definition.criticality if definition.criticality >= 0.8 else 0.0

    def _determinate_gain(
        self, state: RequirementState, context: PolicyContext
    ) -> float:
        return sum(
            context.field_schema[field_id].criticality
            for field_id in self.answer_model.uniquely_determined_values(state)
            if context.field_schema[field_id].criticality >= 0.8
        )

    def _branches(
        self, state: RequirementState, field_id: str
    ) -> list[WeightedAnswer]:
        return self.answer_model.answers_for(state, field_id)

    def _transition(
        self, state: RequirementState, field_id: str, value: Any
    ) -> RequirementState | None:
        next_state = state.apply_answer(
            field_id,
            FieldValue(value=value, status=FieldStatus.OBSERVED, source="ccla_model"),
        )
        result = self.engine.check_satisfiability(next_state)
        if result.satisfiable is not True:
            return None
        return next_state

    def _best_direct_gain(
        self, state: RequirementState, context: PolicyContext
    ) -> float:
        return max(
            (
                self._direct_gain(state, field_id, context)
                for field_id in _candidate_ids(state, context)
            ),
            default=0.0,
        )


class CCLAOneStepPolicy(_CCLABasePolicy):
    name = "ccla_one_step"

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        scored: list[tuple[str, float, dict[str, float]]] = []
        for field_id in _candidate_ids(state, context):
            branches = self._branches(state, field_id)
            if not branches:
                continue
            direct_gain = self._direct_gain(state, field_id, context)
            penalty = self.lambda_cost * context.field_schema[field_id].answer_cost
            scored.append(
                (
                    field_id,
                    direct_gain - penalty,
                    {
                        "immediate_resolved_gain": direct_gain,
                        "lookahead_gain": 0.0,
                        "answer_cost_penalty": penalty,
                        "solver_cost_penalty": 0.0,
                        "counterfactual_calls": 0.0,
                    },
                )
            )
        if not scored:
            return PolicyDecision(field_id=None)
        field_id, total, terms = min(
            scored,
            key=lambda item: (
                -item[1],
                context.field_schema[item[0]].answer_cost,
                item[0],
            ),
        )
        return PolicyDecision(field_id=field_id, utility_terms={**terms, "total": total})


class CCLATwoStepPolicy(_CCLABasePolicy):
    name = "ccla_two_step"

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        scored: list[tuple[str, float, dict[str, float]]] = []
        for field_id in _candidate_ids(state, context):
            branches = self._branches(state, field_id)
            if not branches:
                continue
            immediate_gain = self._direct_gain(state, field_id, context)
            lookahead_gain = 0.0
            calls = 0
            for branch in branches:
                calls += 1
                next_state = self._transition(state, field_id, branch.value)
                if next_state is None:
                    continue
                lookahead_gain += branch.probability * (
                    self._determinate_gain(next_state, context)
                    + self._best_direct_gain(next_state, context)
                )
            answer_penalty = self.lambda_cost * context.field_schema[field_id].answer_cost
            solver_penalty = self.lambda_solver * calls
            total = immediate_gain + lookahead_gain - answer_penalty - solver_penalty
            scored.append(
                (
                    field_id,
                    total,
                    {
                        "immediate_resolved_gain": immediate_gain,
                        "lookahead_gain": lookahead_gain,
                        "answer_cost_penalty": answer_penalty,
                        "solver_cost_penalty": solver_penalty,
                        "counterfactual_calls": float(calls),
                    },
                )
            )
        if not scored:
            return PolicyDecision(field_id=None)
        field_id, total, terms = min(
            scored,
            key=lambda item: (
                -item[1],
                context.field_schema[item[0]].answer_cost,
                int(item[2]["counterfactual_calls"]),
                item[0],
            ),
        )
        return PolicyDecision(field_id=field_id, utility_terms={**terms, "total": total})
