from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldValue,
    RequirementState,
)
from paper1_active_req.generator.complete_specs import RequirementTask
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import AcquisitionPolicy, PolicyDecision
from paper1_active_req.solver.engine import ConstraintEngine


class PolicyContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_schema: dict[str, FieldDefinition]
    budget: int
    step_index: int


class EpisodeStep(BaseModel):
    step_index: int
    field_id: str
    utility_terms: dict[str, float]
    answer: FieldValue
    satisfiable_after_answer: bool | None
    unsat_core_after_answer: list[str] = Field(default_factory=list)


class EpisodeResult(BaseModel):
    task_id: str
    policy_name: str
    initial_state: RequirementState
    final_state: RequirementState
    steps: list[EpisodeStep]
    solver_calls: int


def run_episode(
    *,
    task: RequirementTask,
    policy: AcquisitionPolicy,
    oracle: SimulatedOracle,
    engine: ConstraintEngine,
    budget: int = 8,
) -> EpisodeResult:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    initial_state = task.initial_state.public_copy()
    state = initial_state.public_copy()
    steps: list[EpisodeStep] = []
    solver_calls = 0

    for step_index in range(budget):
        solver_result = engine.check(state)
        solver_calls += 1
        context = PolicyContext(
            field_schema=engine.schema,
            budget=budget,
            step_index=step_index,
        )
        decision = policy.select(
            state.public_copy(), solver_result, context
        )
        if decision is None:
            break
        if not isinstance(decision, PolicyDecision):
            decision = PolicyDecision.model_validate(decision)
        if decision.field_id is None:
            break
        if decision.field_id not in state.queryable_field_ids():
            raise ValueError(
                f"policy selected non-queryable field: {decision.field_id}"
            )

        answer = oracle.answer(decision.field_id)
        state = state.apply_answer(decision.field_id, answer)
        after = engine.check(state)
        solver_calls += 1
        steps.append(
            EpisodeStep(
                step_index=step_index,
                field_id=decision.field_id,
                utility_terms=decision.utility_terms,
                answer=answer,
                satisfiable_after_answer=after.satisfiable,
                unsat_core_after_answer=after.unsat_core,
            )
        )
        if not state.queryable_field_ids():
            break

    if budget == 0:
        engine.check(state)
        solver_calls += 1
    return EpisodeResult(
        task_id=task.task_id,
        policy_name=policy.name,
        initial_state=initial_state,
        final_state=state,
        steps=steps,
        solver_calls=solver_calls,
    )
