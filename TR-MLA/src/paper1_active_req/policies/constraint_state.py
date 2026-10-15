from __future__ import annotations

from pydantic import BaseModel, Field

from paper1_active_req.domain.models import RequirementState
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.solver.engine import ConstraintEngine, SolverResult

from .base import EmpiricalAnswerModel, PolicyDecision
from .baselines import _candidate_ids, _state_width


class UtilityWeights(BaseModel):
    domain_reduction: float = Field(ge=0.0)
    criticality: float = Field(ge=0.0)
    conflict: float = Field(ge=0.0)
    graph_degree: float = Field(ge=0.0)
    answer_cost: float = Field(ge=0.0)


class CandidateScore(BaseModel):
    field_id: str
    raw_terms: dict[str, float]
    normalized_terms: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0


class ConstraintStatePolicy:
    name = "constraint_state_full"

    def __init__(
        self,
        *,
        engine: ConstraintEngine,
        empirical_model: EmpiricalAnswerModel,
        weights: UtilityWeights,
    ) -> None:
        self.engine = engine
        self.empirical_model = empirical_model
        self.weights = weights

    def select(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> PolicyDecision:
        scores = self.score_candidates(state, solver_result, context)
        if not scores:
            return PolicyDecision(field_id=None)
        selected = min(scores, key=lambda item: (-item.total, item.field_id))
        return PolicyDecision(
            field_id=selected.field_id,
            utility_terms={
                **selected.normalized_terms,
                "total": selected.total,
            },
        )

    def score_candidates(
        self,
        state: RequirementState,
        solver_result: SolverResult,
        context: PolicyContext,
    ) -> list[CandidateScore]:
        current_width = _state_width(solver_result, context)
        conflict_fields = self.engine.fields_for_rules(
            solver_result.unsat_core
        )
        scores: list[CandidateScore] = []
        for field_id in _candidate_ids(state, context):
            answers = self.empirical_model.answers_for(field_id)
            if not answers:
                continue
            total_probability = sum(
                answer.probability for answer in answers
            )
            expected_width = 0.0
            valid_probability = 0.0
            for answer in answers:
                probability = answer.probability / total_probability
                counterfactual = self.engine.counterfactual(
                    state, field_id, answer.value
                )
                if counterfactual.satisfiable is True:
                    expected_width += probability * _state_width(
                        counterfactual, context
                    )
                    valid_probability += probability
                else:
                    expected_width += probability * current_width
            domain_reduction = (
                current_width - expected_width
                if valid_probability > 0.0
                else 0.0
            )
            definition = context.field_schema[field_id]
            scores.append(
                CandidateScore(
                    field_id=field_id,
                    raw_terms={
                        "domain_reduction": domain_reduction,
                        "criticality": definition.criticality,
                        "conflict": float(field_id in conflict_fields),
                        "graph_degree": self.engine.rule_graph_degree(
                            field_id
                        ),
                        "answer_cost": definition.answer_cost,
                    },
                )
            )
        self._normalize_and_total(scores)
        return scores

    def _normalize_and_total(
        self, scores: list[CandidateScore]
    ) -> None:
        term_names = [
            "domain_reduction",
            "criticality",
            "conflict",
            "graph_degree",
            "answer_cost",
        ]
        for term_name in term_names:
            values = [item.raw_terms[term_name] for item in scores]
            normalized = _minmax(values)
            for item, value in zip(scores, normalized, strict=True):
                item.normalized_terms[term_name] = value

        for item in scores:
            terms = item.normalized_terms
            item.total = (
                self.weights.domain_reduction
                * terms["domain_reduction"]
                + self.weights.criticality * terms["criticality"]
                + self.weights.conflict * terms["conflict"]
                + self.weights.graph_degree * terms["graph_degree"]
                - self.weights.answer_cost * terms["answer_cost"]
            )


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lower = min(values)
    upper = max(values)
    if upper == lower:
        return [0.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]
