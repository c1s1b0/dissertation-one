from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
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
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch
from paper1_active_req.solver.engine import SolverResult


_RELATION_TARGET_IDS = (
    "available_width_mm",
    "mounting_clearance_mm",
    "service_clearance_mm",
)


def _hidden() -> FieldValue:
    return FieldValue(value=None, status=FieldStatus.HIDDEN, source="public-fixture")


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
        confidence=max(probability for _, probability in branches),
        factor_ids=factor_ids,
    )


@dataclass
class PublicRelationModel:
    """仅依据公开 state 暴露 clearance 关系后验的受控模型。"""

    relation_config: object = field(
        default_factory=lambda: SimpleNamespace(engineering_tolerances={})
    )
    factors_by_target: dict[str, tuple[object, ...]] = field(
        default_factory=lambda: {
            field_id: (object(),) for field_id in _RELATION_TARGET_IDS
        }
    )

    def posterior(self, state: RequirementState, field_id: str) -> FieldPosterior:
        if field_id in _RELATION_TARGET_IDS:
            if all(
                state.values[item].status == FieldStatus.OBSERVED
                for item in ("side_clearance_mm", "bridge_query")
            ):
                return _posterior(
                    field_id, ("resolved", 1.0), factor_ids=("clearance_affine",)
                )
            return _posterior(field_id, ("low", 0.5), ("high", 0.5))
        return _posterior(field_id, ("low", 0.5), ("high", 0.5))


@dataclass
class StateDependentNoFactorModel:
    """伪装为无因子边际模型、实际却依赖公开 state 的攻击模型。"""

    relation_config: object = field(
        default_factory=lambda: SimpleNamespace(engineering_tolerances={})
    )
    factors_by_target: dict[str, tuple[object, ...]] = field(default_factory=dict)

    def posterior(self, state: RequirementState, field_id: str) -> FieldPosterior:
        if state.values["side_clearance_mm"].status == FieldStatus.OBSERVED:
            return _posterior(field_id, ("high", 1.0))
        return _posterior(field_id, ("low", 0.5), ("high", 0.5))


@dataclass
class PublicEngine:
    schema: dict[str, FieldDefinition]

    def check_satisfiability(self, state: RequirementState) -> SolverResult:
        del state
        return SolverResult(satisfiable=True)


def _schema() -> dict[str, FieldDefinition]:
    return {
        "side_clearance_mm": FieldDefinition(
            field_id="side_clearance_mm",
            kind=FieldKind.CATEGORICAL,
            categories=("low", "high"),
            criticality=0.1,
            answer_cost=1.0,
        ),
        "bridge_query": FieldDefinition(
            field_id="bridge_query",
            kind=FieldKind.CATEGORICAL,
            categories=("low", "high"),
            criticality=0.3,
            answer_cost=2.0,
        ),
        "z_alternative": FieldDefinition(
            field_id="z_alternative",
            kind=FieldKind.CATEGORICAL,
            categories=("low", "high"),
            criticality=0.9,
            answer_cost=1.0,
        ),
        **{
            field_id: FieldDefinition(
                field_id=field_id,
                kind=FieldKind.CATEGORICAL,
                categories=("low", "high"),
                criticality=1.0,
                answer_cost=10.0,
                queryable=True,
            )
            for field_id in _RELATION_TARGET_IDS
        },
    }


def _train_specs(schema: dict[str, FieldDefinition]) -> list[CompleteSpec]:
    def values(answer: str) -> dict[str, FieldValue]:
        return {
            field_id: FieldValue(
                value=answer,
                status=FieldStatus.OBSERVED,
                source="train-fixture",
            )
            for field_id in schema
        }

    return [
        CompleteSpec(spec_id="train-1", values=values("low"), split="train"),
        CompleteSpec(spec_id="train-2", values=values("high"), split="train"),
    ]


def policy_fixture():
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadNoRelationsPolicy,
        TRMLAPolicy,
        TypedRelationalMyopicPolicy,
    )

    schema = _schema()
    state = RequirementState(values={field_id: _hidden() for field_id in schema})
    context = PolicyContext(field_schema=schema, budget=2, step_index=0)
    engine = PublicEngine(schema)
    full = TRMLAPolicy(
        initial_state=state, model=PublicRelationModel(), engine=engine, lambda_cost=0.2
    )
    myopic = TypedRelationalMyopicPolicy(
        initial_state=state, model=PublicRelationModel(), engine=engine, lambda_cost=0.2
    )
    no_relations = MarginalLookaheadNoRelationsPolicy.from_training_specs(
        initial_state=state,
        training_specs=_train_specs(schema),
        schema=schema,
        relation_config=load_v3_relation_config(Path("configs/v3_relations.yaml")),
        engine=engine,
        lambda_cost=0.2,
    )
    return full, myopic, no_relations, state, SolverResult(satisfiable=True), context


def test_trmla_selects_hidden_clearance_pivot_only_when_it_has_positive_marginal_value():
    """若删去关系二步增益，此选择应退化为 z_alternative。"""
    full, myopic, no_relations, state, solver_result, context = policy_fixture()

    full_decision = full.select(state, solver_result, context)
    myopic_decision = myopic.select(state, solver_result, context)
    no_relations_decision = no_relations.select(state, solver_result, context)

    assert full_decision.field_id == "side_clearance_mm"
    assert full_decision.utility_terms["first_step_gain"] == 0.0
    assert full_decision.utility_terms["second_step_gain"] > 0.0
    assert myopic_decision.field_id == "z_alternative"
    assert myopic_decision.utility_terms["second_step_gain"] == 0.0
    assert no_relations_decision.field_id == "z_alternative"


def test_myopic_policy_sets_second_step_gain_to_exactly_zero():
    """若重新启用第二步收益，此断言会保护消融定义。"""
    _, myopic, _, state, solver_result, context = policy_fixture()

    decision = myopic.select(state, solver_result, context)

    assert decision.utility_terms["second_step_gain"] == 0.0
    assert decision.utility_terms["marginal_score"] == pytest.approx(
        decision.utility_terms["first_step_gain"]
        - 0.2 * decision.utility_terms["normalized_answer_cost"]
    )


def test_no_relations_policy_has_no_public_model_injection_path():
    """若可注入伪无因子状态模型，消融就会泄漏关系信息。"""
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadNoRelationsPolicy,
    )

    state = RequirementState(values={field_id: _hidden() for field_id in _schema()})
    fake_model = StateDependentNoFactorModel()
    changed_state = state.apply_answer(
        "side_clearance_mm",
        FieldValue(value="low", status=FieldStatus.OBSERVED, source="public-fixture"),
    )

    assert "model" not in inspect.signature(MarginalLookaheadNoRelationsPolicy).parameters
    assert fake_model.posterior(state, "bridge_query") != fake_model.posterior(
        changed_state, "bridge_query"
    )
    with pytest.raises(TypeError):
        MarginalLookaheadNoRelationsPolicy(
            initial_state=state,
            model=fake_model,
            engine=PublicEngine(_schema()),
            lambda_cost=0.0,
        )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_no_relations_policy_rejects_non_train_specs(split: str):
    """若允许 validation/test 规格进入，训练边际比较会泄漏。"""
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadNoRelationsPolicy,
    )

    schema = _schema()
    state = RequirementState(values={field_id: _hidden() for field_id in schema})
    contaminated = _train_specs(schema)
    contaminated[-1] = contaminated[-1].model_copy(update={"split": split})

    with pytest.raises(ValueError, match="split=train"):
        MarginalLookaheadNoRelationsPolicy.from_training_specs(
            initial_state=state,
            training_specs=contaminated,
            schema=schema,
            relation_config=load_v3_relation_config(Path("configs/v3_relations.yaml")),
            engine=PublicEngine(schema),
            lambda_cost=0.0,
        )


def test_no_relations_policy_uses_state_independent_train_marginals_without_factor_ids():
    """若 posterior 开始依赖公开状态或携带因子 ID，此测试会失败。"""
    from paper1_active_req.policies.marginal_lookahead import (
        MarginalLookaheadNoRelationsPolicy,
    )

    schema = _schema()
    state = RequirementState(values={field_id: _hidden() for field_id in schema})
    policy = MarginalLookaheadNoRelationsPolicy.from_training_specs(
        initial_state=state,
        training_specs=_train_specs(schema),
        schema=schema,
        relation_config=load_v3_relation_config(Path("configs/v3_relations.yaml")),
        engine=PublicEngine(schema),
        lambda_cost=0.0,
    )
    changed_public_state = state.apply_answer(
        "side_clearance_mm",
        FieldValue(value="low", status=FieldStatus.OBSERVED, source="public-fixture"),
    )

    assert policy.evaluator.model.factors_by_target == {}
    for field_id, posterior in policy.evaluator.model.marginal_posteriors.items():
        assert posterior.factor_ids == ()
        assert policy.evaluator.model.posterior(state, field_id) == posterior
        assert policy.evaluator.model.posterior(changed_public_state, field_id) == posterior


def test_v3_policy_interfaces_cannot_receive_audit_metadata():
    from paper1_active_req.policies.marginal_lookahead import TRMLAPolicy

    assert "audit" not in inspect.signature(TRMLAPolicy).parameters
    assert "ground_truth" not in inspect.signature(TRMLAPolicy).parameters
    assert "audit" not in inspect.signature(TRMLAPolicy.select).parameters
    assert "ground_truth" not in inspect.signature(TRMLAPolicy.select).parameters
