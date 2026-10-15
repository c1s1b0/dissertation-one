from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldKind,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch
from paper1_active_req.solver.engine import SolverResult


def _hidden() -> FieldValue:
    return FieldValue(value=None, status=FieldStatus.HIDDEN, source="public-fixture")


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED, source="public-fixture")


def _posterior(
    field_id: str,
    *branches: tuple[object, float],
    factor_ids: tuple[str, ...] = (),
) -> FieldPosterior:
    return FieldPosterior(
        field_id=field_id,
        branches=tuple(
            OutcomeBranch(value=value, probability=probability)
            for value, probability in branches
        ),
        confidence=max((probability for _, probability in branches), default=0.0),
        factor_ids=factor_ids,
    )


@dataclass
class ControlledPosteriorModel:
    posteriors: dict[str, FieldPosterior | None]
    relation_config: object = field(
        default_factory=lambda: SimpleNamespace(engineering_tolerances={})
    )

    def posterior(
        self, state: RequirementState, field_id: str
    ) -> FieldPosterior | None:
        del state
        return self.posteriors[field_id]


@dataclass
class ControlledEngine:
    unsatisfiable_answers: set[object] = field(default_factory=set)
    calls: list[RequirementState] = field(default_factory=list)

    def check_satisfiability(self, state: RequirementState) -> SolverResult:
        self.calls.append(state.public_copy())
        answer = next(
            (
                value.value
                for value in state.values.values()
                if value.source == "marginal_lookahead"
            ),
            None,
        )
        return SolverResult(satisfiable=answer not in self.unsatisfiable_answers)


@dataclass
class StateSensitivePosteriorModel:
    relation_config: object = field(
        default_factory=lambda: SimpleNamespace(engineering_tolerances={})
    )

    def posterior(
        self, state: RequirementState, field_id: str
    ) -> FieldPosterior:
        if field_id == "bridge_b" and state.values["pivot_a"].status == FieldStatus.OBSERVED:
            return _posterior("bridge_b", ("low", 1.0), factor_ids=("a_to_b",))
        if field_id == "bridge_b":
            return _posterior("bridge_b", ("low", 0.5), ("high", 0.5))
        return _posterior(field_id, ("low", 0.5), ("high", 0.5))


def _schema(**weights_and_costs: tuple[float, float]) -> dict[str, FieldDefinition]:
    return {
        field_id: FieldDefinition(
            field_id=field_id,
            kind=FieldKind.CATEGORICAL,
            categories=("low", "high"),
            criticality=weight,
            answer_cost=cost,
        )
        for field_id, (weight, cost) in weights_and_costs.items()
    }


def _context(schema: dict[str, FieldDefinition]) -> PolicyContext:
    return PolicyContext(field_schema=schema, budget=2, step_index=0)


def test_already_inferred_field_contributes_zero_additional_gain():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(side_clearance_mm=(0.2, 1.0), available_width_mm=(0.8, 2.0))
    state = RequirementState(
        values={"side_clearance_mm": _hidden(), "available_width_mm": _hidden()}
    )
    model = ControlledPosteriorModel(
        {
            "side_clearance_mm": _posterior(
                "side_clearance_mm", ("low", 0.5), ("high", 0.5)
            ),
            "available_width_mm": _posterior(
                "available_width_mm", ("high", 1.0), factor_ids=("width_relation",)
            ),
        }
    )
    evaluator = ResolvedScoreEvaluator(state, schema, model)
    scorer = MarginalLookaheadScorer(evaluator, ControlledEngine(), lambda_cost=0.0)

    score = scorer.score_action(state, "side_clearance_mm", _context(schema))

    assert "available_width_mm" not in score.second_step_newly_resolved
    assert score.first_step_gain + score.second_step_gain <= 1.0
    assert score.first_step_gain == pytest.approx(0.0)


def test_two_step_marginal_gain_telescopes_exactly():
    from paper1_active_req.policies.marginal_lookahead import ResolvedScoreEvaluator

    schema = _schema(alpha=(0.8, 1.0), beta=(0.9, 1.0))
    s0 = RequirementState(values={"alpha": _hidden(), "beta": _hidden()})
    s1 = s0.apply_answer("alpha", _observed("low"))
    s2 = s1.apply_answer("beta", _observed("high"))
    model = ControlledPosteriorModel(
        {
            "alpha": _posterior("alpha", ("low", 0.5), ("high", 0.5)),
            "beta": _posterior("beta", ("low", 0.5), ("high", 0.5)),
        }
    )
    evaluator = ResolvedScoreEvaluator(s0, schema, model)

    left = evaluator.resolved_score(s1) - evaluator.resolved_score(s0)
    left += evaluator.resolved_score(s2) - evaluator.resolved_score(s1)
    right = evaluator.resolved_score(s2) - evaluator.resolved_score(s0)

    assert left == pytest.approx(right)
    assert evaluator.resolved_score(s2) == pytest.approx(1.0)


def test_low_criticality_field_remains_queryable_but_has_zero_direct_resolved_gain():
    """低于 0.8 的真实量级字段不能改变 resolved score 分母或直接收益。"""
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(side_clearance_mm=(0.6, 1.0), available_width_mm=(1.0, 1.0))
    state = RequirementState(
        values={"side_clearance_mm": _hidden(), "available_width_mm": _hidden()}
    )
    evaluator = ResolvedScoreEvaluator(
        state,
        schema,
        ControlledPosteriorModel(
            {
                "side_clearance_mm": _posterior(
                    "side_clearance_mm", ("low", 0.5), ("high", 0.5)
                ),
                "available_width_mm": _posterior(
                    "available_width_mm", ("low", 0.5), ("high", 0.5)
                ),
            }
        ),
    )
    scorer = MarginalLookaheadScorer(evaluator, ControlledEngine(), lambda_cost=0.0)

    assert "side_clearance_mm" in state.queryable_field_ids()
    assert evaluator.resolved_score(state.apply_answer("side_clearance_mm", _observed("low"))) == 0.0
    assert scorer.score_action(state, "side_clearance_mm", _context(schema)).first_step_gain == 0.0


def test_initially_reliable_fields_are_excluded_from_frozen_denominator():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(initially_reliable=(0.8, 1.0), question=(0.9, 1.0))
    state = RequirementState(
        values={"initially_reliable": _hidden(), "question": _hidden()}
    )
    model = ControlledPosteriorModel(
        {
            "initially_reliable": _posterior("initially_reliable", ("low", 1.0)),
            "question": _posterior("question", ("low", 0.5), ("high", 0.5)),
        }
    )
    evaluator = ResolvedScoreEvaluator(state, schema, model)
    scorer = MarginalLookaheadScorer(evaluator, ControlledEngine(), lambda_cost=0.0)

    score = scorer.score_action(state, "question", _context(schema))

    assert evaluator.resolved_score(state) == pytest.approx(0.8 / 1.7)
    assert score.first_step_gain == pytest.approx(0.9 / 1.7)


def test_first_step_inference_is_not_recounted_when_second_step_queries_c():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(pivot_a=(0.8, 1.0), bridge_b=(0.9, 1.0), query_c=(1.0, 1.0))
    s0 = RequirementState(
        values={"pivot_a": _hidden(), "bridge_b": _hidden(), "query_c": _hidden()}
    )
    model = StateSensitivePosteriorModel()
    evaluator = ResolvedScoreEvaluator(s0, schema, model)
    scorer = MarginalLookaheadScorer(evaluator, ControlledEngine(), lambda_cost=0.0)

    score = scorer.score_action(s0, "pivot_a", _context(schema))
    s1 = s0.apply_answer("pivot_a", _observed("low"))
    s2 = s1.apply_answer("query_c", _observed("low"))

    assert score.first_step_newly_resolved == ("pivot_a", "bridge_b")
    assert score.second_step_newly_resolved == ("query_c",)
    assert "bridge_b" not in score.second_step_newly_resolved
    assert score.first_step_gain + score.second_step_gain == pytest.approx(1.0)
    assert score.first_step_gain + score.second_step_gain == pytest.approx(
        evaluator.resolved_score(s2) - evaluator.resolved_score(s0)
    )


def test_unsatisfiable_counterfactual_branches_are_removed_and_renormalized():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(question=(1.0, 1.0))
    state = RequirementState(values={"question": _hidden()})
    model = ControlledPosteriorModel(
        {"question": _posterior("question", ("low", 0.2), ("high", 0.8), factor_ids=("q",))}
    )
    engine = ControlledEngine(unsatisfiable_answers={"high"})
    scorer = MarginalLookaheadScorer(
        ResolvedScoreEvaluator(state, schema, model), engine, lambda_cost=0.0
    )

    score = scorer.score_action(state, "question", _context(schema))

    assert score.counterfactual_branch_count == 2
    assert score.feasible_branch_count == 1
    assert score.first_step_gain == pytest.approx(1.0)
    assert len(engine.calls) == 2


def test_missing_posterior_does_not_fall_back_to_any_other_distribution():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(question=(1.0, 1.0))
    state = RequirementState(values={"question": _hidden()})
    model = ControlledPosteriorModel({"question": None})
    model.marginal_posteriors = {"question": _posterior("question", ("low", 1.0))}
    engine = ControlledEngine()
    scorer = MarginalLookaheadScorer(
        ResolvedScoreEvaluator(state, schema, model), engine, lambda_cost=0.0
    )

    score = scorer.score_action(state, "question", _context(schema))

    assert score.total == pytest.approx(0.0)
    assert score.counterfactual_branch_count == 0
    assert engine.calls == []


def test_equal_scores_use_field_id_tie_break_and_normalize_answer_cost():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadScorer,
        ResolvedScoreEvaluator,
    )

    schema = _schema(alpha=(0.5, 2.0), beta=(0.5, 2.0), expensive_reference=(0.0, 4.0))
    state = RequirementState(values={"alpha": _hidden(), "beta": _hidden(), "expensive_reference": _observed("low")})
    model = ControlledPosteriorModel(
        {
            "alpha": _posterior(
                "alpha", ("low", 0.5), ("high", 0.5), factor_ids=("a",)
            ),
            "beta": _posterior(
                "beta", ("low", 0.5), ("high", 0.5), factor_ids=("b",)
            ),
            "expensive_reference": _posterior("expensive_reference", ("low", 1.0)),
        }
    )
    scorer = MarginalLookaheadScorer(
        ResolvedScoreEvaluator(state, schema, model), ControlledEngine(), lambda_cost=0.4
    )

    score = scorer.score_action(state, "alpha", _context(schema))

    assert scorer.select_action(state, _context(schema)) == "alpha"
    assert score.normalized_answer_cost == pytest.approx(0.5)
    assert score.total == pytest.approx(
        score.first_step_gain + score.second_step_gain - 0.2
    )
    assert score.relation_factor_ids == ("a", "b")
    assert {"ground_truth", "audit"}.isdisjoint(score.__dataclass_fields__)
