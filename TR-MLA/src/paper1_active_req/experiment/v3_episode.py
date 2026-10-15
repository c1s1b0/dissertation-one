from __future__ import annotations

from pydantic import BaseModel, Field

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import AcquisitionPolicy, PolicyDecision
from paper1_active_req.solver.engine import ConstraintEngine


class V3PolicyDecision(PolicyDecision):
    """不含 audit 或 task 真值的 v3 策略决策元数据。"""

    relation_factor_ids: tuple[str, ...] = ()
    resolved_before: tuple[str, ...] = ()
    marginal_score: float | None = None


class V3EpisodeStep(BaseModel):
    step_index: int
    field_id: str
    utility_terms: dict[str, float] = Field(default_factory=dict)
    answer: FieldValue
    satisfiable_after_answer: bool | None
    unsat_core_after_answer: list[str] = Field(default_factory=list)
    relation_factor_ids: tuple[str, ...] = ()
    resolved_before: tuple[str, ...] = ()
    resolved_after: tuple[str, ...] = ()
    marginal_score: float | None = None


class V3EpisodeResult(BaseModel):
    task_id: str
    policy_name: str
    initial_state: RequirementState
    final_state: RequirementState
    steps: list[V3EpisodeStep]
    solver_calls: int


def _public_resolved_ids(
    policy: AcquisitionPolicy,
    state: RequirementState,
) -> tuple[str, ...]:
    resolver = getattr(policy, "resolved_field_ids", None)
    if callable(resolver):
        return tuple(resolver(state.public_copy()))
    return tuple(
        field_id
        for field_id, value in state.values.items()
        if value.status == FieldStatus.OBSERVED and value.value is not None
    )


def run_v3_episode(
    *,
    task: V3RequirementTask,
    policy: AcquisitionPolicy,
    oracle: SimulatedOracle,
    engine: ConstraintEngine,
    budget: int = 8,
) -> V3EpisodeResult:
    """运行 v3 episode；策略永远只接收 public_copy，真值留在 Oracle 内。"""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    initial_state = task.initial_state.public_copy()
    state = initial_state.public_copy()
    steps: list[V3EpisodeStep] = []
    solver_calls = 0

    for step_index in range(budget):
        solver_result = engine.check(state)
        solver_calls += 1
        context = PolicyContext(
            field_schema=engine.schema,
            budget=budget,
            step_index=step_index,
        )
        resolved_before = _public_resolved_ids(policy, state)
        decision = policy.select(state.public_copy(), solver_result, context)
        if decision is None:
            break
        if not isinstance(decision, PolicyDecision):
            decision = PolicyDecision.model_validate(decision)
        if decision.field_id is None:
            break
        if decision.field_id not in state.queryable_field_ids():
            raise ValueError(f"policy selected non-queryable field: {decision.field_id}")

        answer = oracle.answer(decision.field_id)
        state = state.apply_answer(decision.field_id, answer)
        after = engine.check(state)
        solver_calls += 1
        if isinstance(decision, V3PolicyDecision):
            relation_factor_ids = decision.relation_factor_ids
            marginal_score = decision.marginal_score
        else:
            relation_factor_ids = ()
            marginal_score = None
        steps.append(
            V3EpisodeStep(
                step_index=step_index,
                field_id=decision.field_id,
                utility_terms=decision.utility_terms,
                answer=answer,
                satisfiable_after_answer=after.satisfiable,
                unsat_core_after_answer=after.unsat_core,
                relation_factor_ids=relation_factor_ids,
                resolved_before=resolved_before,
                resolved_after=_public_resolved_ids(policy, state),
                marginal_score=marginal_score,
            )
        )
        if not state.queryable_field_ids():
            break

    if budget == 0:
        engine.check(state)
        solver_calls += 1
    return V3EpisodeResult(
        task_id=task.task_id,
        policy_name=policy.name,
        initial_state=initial_state,
        final_state=state,
        steps=steps,
        solver_calls=solver_calls,
    )
