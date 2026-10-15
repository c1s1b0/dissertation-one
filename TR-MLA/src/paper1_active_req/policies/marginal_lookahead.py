from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.experiment.v3_episode import V3PolicyDecision
from paper1_active_req.evaluation.v3_protocol import frozen_critical_field_ids
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.policies.base import PolicyDecision
from paper1_active_req.relations.config import V3RelationConfig
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.relations.posterior import FieldPosterior
from paper1_active_req.solver.engine import ConstraintEngine


@dataclass(frozen=True)
class MarginalActionScore:
    """只保存公开的边际得分审计元数据。"""

    field_id: str
    total: float
    first_step_gain: float
    second_step_gain: float
    normalized_answer_cost: float
    first_step_newly_resolved: tuple[str, ...]
    second_step_newly_resolved: tuple[str, ...]
    counterfactual_branch_count: int
    feasible_branch_count: int
    first_step_branch_count: int
    first_step_feasible_branch_count: int
    second_step_branch_count: int
    second_step_feasible_branch_count: int
    relation_factor_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Transition:
    state: RequirementState
    probability: float


@dataclass(frozen=True)
class _StepEvaluation:
    gain: float
    newly_resolved: tuple[str, ...]
    branch_count: int
    feasible_branch_count: int
    relation_factor_ids: tuple[str, ...]


class ResolvedScoreEvaluator:
    """以初始公开状态冻结的关键字段权重计算已解析比例。"""

    def __init__(
        self,
        initial_state: RequirementState,
        schema: dict[str, Any],
        model: TypedRelationalModel,
        threshold: float = 0.95,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.initial_state = initial_state.public_copy()
        self.schema = dict(schema)
        self.model = model
        self.threshold = threshold
        self._initial_unresolved_ids = frozen_critical_field_ids(
            self.initial_state, self.schema
        )
        self._denominator = sum(
            self.schema[field_id].criticality
            for field_id in self._initial_unresolved_ids
        )

    def resolved_score(self, state: RequirementState) -> float:
        if self._denominator == 0.0:
            return 0.0
        return sum(
            self.schema[field_id].criticality
            for field_id in self.resolved_field_ids(state)
        ) / self._denominator

    def newly_resolved(
        self, before: RequirementState, after: RequirementState
    ) -> tuple[str, ...]:
        before_ids = set(self.resolved_field_ids(before))
        return tuple(
            field_id
            for field_id in self.resolved_field_ids(after)
            if field_id not in before_ids
        )

    def resolved_field_ids(self, state: RequirementState) -> tuple[str, ...]:
        resolved: list[str] = []
        for field_id in self._initial_unresolved_ids:
            if self._is_resolved(state, field_id):
                resolved.append(field_id)
        return tuple(resolved)

    def _is_resolved(self, state: RequirementState, field_id: str) -> bool:
        value = state.values.get(field_id)
        if value is None:
            return False
        if value.status == FieldStatus.OBSERVED and value.value is not None:
            return True
        posterior = self.model.posterior(state, field_id)
        return posterior is not None and self._posterior_is_reliable(field_id, posterior)

    def _posterior_is_reliable(
        self, field_id: str, posterior: FieldPosterior
    ) -> bool:
        if posterior.confidence >= self.threshold:
            return True
        tolerances = getattr(
            getattr(self.model, "relation_config", None),
            "engineering_tolerances",
            {},
        )
        tolerance = tolerances.get(field_id)
        return (
            tolerance is not None
            and posterior.interval_low is not None
            and posterior.interval_high is not None
            and posterior.interval_high - posterior.interval_low <= tolerance
        )


class MarginalLookaheadScorer:
    """严格以一次解析增量计分的两步公开状态前瞻。"""

    def __init__(
        self,
        evaluator: ResolvedScoreEvaluator,
        engine: ConstraintEngine,
        *,
        lambda_cost: float,
    ) -> None:
        if lambda_cost < 0.0:
            raise ValueError("lambda_cost must be non-negative")
        self.evaluator = evaluator
        self.engine = engine
        self.lambda_cost = lambda_cost

    def score_action(
        self,
        state: RequirementState,
        field_id: str,
        context: PolicyContext,
    ) -> MarginalActionScore:
        if field_id not in context.field_schema:
            raise KeyError(field_id)
        first_transitions, first_branch_count, first_factors = self._transitions(
            state, field_id
        )
        if not first_transitions:
            return self._empty_score(field_id)

        first_gain = 0.0
        first_new: list[str] = []
        second_gain = 0.0
        second_new: list[str] = []
        second_branch_count = 0
        second_feasible_count = 0
        factor_ids = list(first_factors)
        before_score = self.evaluator.resolved_score(state)

        for transition in first_transitions:
            after_first = transition.state
            first_gain += transition.probability * (
                self.evaluator.resolved_score(after_first) - before_score
            )
            first_new.extend(self.evaluator.newly_resolved(state, after_first))
            second = self._best_second_step(after_first, context)
            second_gain += transition.probability * second.gain
            second_new.extend(second.newly_resolved)
            second_branch_count += second.branch_count
            second_feasible_count += second.feasible_branch_count
            factor_ids.extend(second.relation_factor_ids)

        cost = self._normalized_answer_cost(field_id, context)
        return MarginalActionScore(
            field_id=field_id,
            total=first_gain + second_gain - self.lambda_cost * cost,
            first_step_gain=first_gain,
            second_step_gain=second_gain,
            normalized_answer_cost=cost,
            first_step_newly_resolved=_ordered_unique(first_new),
            second_step_newly_resolved=_ordered_unique(second_new),
            counterfactual_branch_count=first_branch_count + second_branch_count,
            feasible_branch_count=len(first_transitions) + second_feasible_count,
            first_step_branch_count=first_branch_count,
            first_step_feasible_branch_count=len(first_transitions),
            second_step_branch_count=second_branch_count,
            second_step_feasible_branch_count=second_feasible_count,
            relation_factor_ids=_ordered_unique(factor_ids),
        )

    def select_action(
        self, state: RequirementState, context: PolicyContext
    ) -> str | None:
        candidates = self._candidate_ids(state, context)
        if not candidates:
            return None
        scores = [self.score_action(state, field_id, context) for field_id in candidates]
        selectable = [score for score in scores if score.counterfactual_branch_count]
        if not selectable:
            return None
        return min(
            selectable,
            key=lambda score: (
                -score.total,
                score.normalized_answer_cost,
                score.field_id,
            ),
        ).field_id

    def _best_second_step(
        self, state: RequirementState, context: PolicyContext
    ) -> _StepEvaluation:
        candidates = self._candidate_ids(state, context)
        if not candidates:
            return _StepEvaluation(0.0, (), 0, 0, ())
        scored = [self._evaluate_one_step(state, field_id) for field_id in candidates]
        return min(
            zip(candidates, scored, strict=True),
            key=lambda item: (-item[1].gain, item[0]),
        )[1]

    def _evaluate_one_step(
        self, state: RequirementState, field_id: str
    ) -> _StepEvaluation:
        transitions, branch_count, factors = self._transitions(state, field_id)
        before_score = self.evaluator.resolved_score(state)
        return _StepEvaluation(
            gain=sum(
                transition.probability
                * (self.evaluator.resolved_score(transition.state) - before_score)
                for transition in transitions
            ),
            newly_resolved=_ordered_unique(
                field_id
                for transition in transitions
                for field_id in self.evaluator.newly_resolved(state, transition.state)
            ),
            branch_count=branch_count,
            feasible_branch_count=len(transitions),
            relation_factor_ids=factors,
        )

    def _transitions(
        self, state: RequirementState, field_id: str
    ) -> tuple[tuple[_Transition, ...], int, tuple[str, ...]]:
        posterior = self.evaluator.model.posterior(state, field_id)
        if posterior is None or not posterior.branches:
            return (), 0, ()
        feasible: list[tuple[RequirementState, float]] = []
        for branch in posterior.branches:
            after = state.apply_answer(
                field_id,
                FieldValue(
                    value=branch.value,
                    status=FieldStatus.OBSERVED,
                    source="marginal_lookahead",
                ),
            )
            if self.engine.check_satisfiability(after).satisfiable is True:
                feasible.append((after, branch.probability))
        total_probability = sum(probability for _, probability in feasible)
        if total_probability == 0.0:
            return (), len(posterior.branches), posterior.factor_ids
        return (
            tuple(
                _Transition(state=after, probability=probability / total_probability)
                for after, probability in feasible
            ),
            len(posterior.branches),
            posterior.factor_ids,
        )

    def _candidate_ids(
        self, state: RequirementState, context: PolicyContext
    ) -> list[str]:
        resolved = set(self.evaluator.resolved_field_ids(state))
        return [
            field_id
            for field_id in state.queryable_field_ids()
            if field_id in context.field_schema
            and context.field_schema[field_id].queryable
            and field_id not in resolved
        ]

    @staticmethod
    def _normalized_answer_cost(field_id: str, context: PolicyContext) -> float:
        maximum = max(
            (definition.answer_cost for definition in context.field_schema.values()),
            default=0.0,
        )
        if maximum == 0.0:
            return 0.0
        return context.field_schema[field_id].answer_cost / maximum

    @staticmethod
    def _empty_score(field_id: str) -> MarginalActionScore:
        return MarginalActionScore(
            field_id=field_id,
            total=0.0,
            first_step_gain=0.0,
            second_step_gain=0.0,
            normalized_answer_cost=0.0,
            first_step_newly_resolved=(),
            second_step_newly_resolved=(),
            counterfactual_branch_count=0,
            feasible_branch_count=0,
            first_step_branch_count=0,
            first_step_feasible_branch_count=0,
            second_step_branch_count=0,
            second_step_feasible_branch_count=0,
            relation_factor_ids=(),
        )


def _ordered_unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


_NO_RELATIONS_CONSTRUCTION_TOKEN = object()


class _MyopicMarginalLookaheadScorer(MarginalLookaheadScorer):
    """与完整 scorer 共享公开候选、成本和平局规则，但严格不计第二步。"""

    def score_action(
        self,
        state: RequirementState,
        field_id: str,
        context: PolicyContext,
    ) -> MarginalActionScore:
        if field_id not in context.field_schema:
            raise KeyError(field_id)
        transitions, branch_count, factor_ids = self._transitions(state, field_id)
        if not transitions:
            return self._empty_score(field_id)
        before_score = self.evaluator.resolved_score(state)
        first_gain = sum(
            transition.probability
            * (self.evaluator.resolved_score(transition.state) - before_score)
            for transition in transitions
        )
        cost = self._normalized_answer_cost(field_id, context)
        return MarginalActionScore(
            field_id=field_id,
            total=first_gain - self.lambda_cost * cost,
            first_step_gain=first_gain,
            second_step_gain=0.0,
            normalized_answer_cost=cost,
            first_step_newly_resolved=_ordered_unique(
                resolved_id
                for transition in transitions
                for resolved_id in self.evaluator.newly_resolved(state, transition.state)
            ),
            second_step_newly_resolved=(),
            counterfactual_branch_count=branch_count,
            feasible_branch_count=len(transitions),
            first_step_branch_count=branch_count,
            first_step_feasible_branch_count=len(transitions),
            second_step_branch_count=0,
            second_step_feasible_branch_count=0,
            relation_factor_ids=factor_ids,
        )


class _MarginalPolicy:
    """将严格边际 scorer 适配为公开状态的 AcquisitionPolicy。"""

    name = "marginal_lookahead"
    scorer_type = MarginalLookaheadScorer

    def __init__(
        self,
        *,
        initial_state: RequirementState,
        model: TypedRelationalModel,
        engine: ConstraintEngine,
        lambda_cost: float,
    ) -> None:
        self.evaluator = ResolvedScoreEvaluator(initial_state, engine.schema, model)
        self.scorer = self.scorer_type(
            self.evaluator,
            engine,
            lambda_cost=lambda_cost,
        )

    def select(
        self,
        state: RequirementState,
        solver_result: Any,
        context: PolicyContext,
    ) -> PolicyDecision:
        del solver_result
        score = self._select_score(state, context)
        if score is None:
            return V3PolicyDecision(field_id=None)
        return V3PolicyDecision(
            field_id=score.field_id,
            utility_terms={
                "marginal_score": score.total,
                "first_step_gain": score.first_step_gain,
                "second_step_gain": score.second_step_gain,
                "normalized_answer_cost": score.normalized_answer_cost,
                "counterfactual_branch_count": float(score.counterfactual_branch_count),
            },
            relation_factor_ids=score.relation_factor_ids,
            resolved_before=self.evaluator.resolved_field_ids(state),
            marginal_score=score.total,
        )

    def resolved_field_ids(self, state: RequirementState) -> tuple[str, ...]:
        """供 v3 episode 在真实 Oracle 回答后记录公开的 resolved 状态。"""
        return self.evaluator.resolved_field_ids(state)

    def _select_score(
        self, state: RequirementState, context: PolicyContext
    ) -> MarginalActionScore | None:
        candidates = self.scorer._candidate_ids(state, context)
        scores = [self.scorer.score_action(state, field_id, context) for field_id in candidates]
        selectable = [score for score in scores if score.counterfactual_branch_count]
        if not selectable:
            return None
        return min(
            selectable,
            key=lambda score: (
                -score.total,
                score.normalized_answer_cost,
                score.field_id,
            ),
        )


class TRMLAPolicy(_MarginalPolicy):
    """使用完整两步严格边际收益的 typed-relational lookahead 策略。"""

    name = "tr_mla"


class TypedRelationalMyopicPolicy(_MarginalPolicy):
    """关系模型消融：第二步增益始终为零。"""

    name = "typed_relational_myopic"
    scorer_type = _MyopicMarginalLookaheadScorer


class MarginalLookaheadNoRelationsPolicy(_MarginalPolicy):
    """无关系消融：只允许训练边际 posterior，拒绝任何关系因子。"""

    name = "marginal_lookahead_no_relations"

    def __init__(
        self,
        *,
        initial_state: RequirementState,
        engine: ConstraintEngine,
        lambda_cost: float,
        _model: TypedRelationalModel,
        _construction_token: object,
    ) -> None:
        if _construction_token is not _NO_RELATIONS_CONSTRUCTION_TOKEN:
            raise TypeError(
                "construct no-relations policy with from_training_specs"
            )
        self._validate_train_marginal_model(_model)
        super().__init__(
            initial_state=initial_state,
            model=_model,
            engine=engine,
            lambda_cost=lambda_cost,
        )

    @staticmethod
    def _validate_train_marginal_model(model: TypedRelationalModel) -> None:
        """验证唯一受控构造所得模型只能提供无因子的训练边际后验。"""
        if type(model) is not TypedRelationalModel:
            raise TypeError("no-relations policy requires TypedRelationalModel")
        if model.factors_by_target:
            raise ValueError("no-relations policy requires no relation factors")
        if any(
            posterior.factor_ids
            for posterior in model.marginal_posteriors.values()
        ):
            raise ValueError("no-relations policy requires factor-free marginals")

    @classmethod
    def from_training_specs(
        cls,
        *,
        initial_state: RequirementState,
        training_specs: Iterable[CompleteSpec],
        schema: dict[str, Any],
        relation_config: V3RelationConfig,
        engine: ConstraintEngine,
        lambda_cost: float,
    ) -> "MarginalLookaheadNoRelationsPolicy":
        """明确以训练集边际分布拟合，不创建或使用关系因子。"""
        specs = tuple(training_specs)
        if any(type(spec) is not CompleteSpec for spec in specs):
            raise TypeError("training_specs must contain CompleteSpec instances")
        if any(spec.split != "train" for spec in specs):
            raise ValueError("no-relations training_specs must all have split=train")
        model = TypedRelationalModel.fit(
            specs,
            schema,
            relation_config,
            use_relations=False,
        )
        return cls(
            initial_state=initial_state,
            engine=engine,
            lambda_cost=lambda_cost,
            _model=model,
            _construction_token=_NO_RELATIONS_CONSTRUCTION_TOKEN,
        )
