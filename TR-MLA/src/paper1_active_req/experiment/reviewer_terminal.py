"""预算终点局部修正诊断：保留前七问，不覆盖TR-MLA原轨迹。"""
from paper1_active_req.domain.models import FieldStatus
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.experiment.v3_episode import V3EpisodeResult, V3EpisodeStep
from paper1_active_req.oracle.simulated import SimulatedOracle


def finish_episode(original, task, last_policy, engine, budget=8):
    if original.task_id != task.task_id or original.initial_state != task.initial_state:
        raise ValueError("terminal diagnostic task binding mismatch")
    if budget < 1 or len(original.steps) > budget:
        raise ValueError("invalid terminal budget")
    if len(original.steps) < budget:
        return original.model_copy(deep=True)
    prefix = [step.model_copy(deep=True) for step in original.steps[:budget-1]]
    state = original.initial_state.public_copy()
    for step in prefix:
        state = state.apply_answer(step.field_id,step.answer)
    context = PolicyContext(field_schema=engine.schema,budget=budget,step_index=budget-1)
    solver = engine.check(state)
    decision = last_policy.select(state.public_copy(),solver,context)
    if decision is not None and decision.field_id is not None:
        if decision.field_id not in state.queryable_field_ids() or not engine.schema[decision.field_id].queryable:
            raise ValueError("terminal policy selected non-queryable field")
        before = tuple(field for field,value in state.values.items() if value.status == FieldStatus.OBSERVED)
        answer = SimulatedOracle(task.ground_truth).answer(decision.field_id)
        state = state.apply_answer(decision.field_id,answer)
        after = engine.check(state)
        prefix.append(V3EpisodeStep(step_index=budget-1,field_id=decision.field_id,answer=answer,
            utility_terms=decision.utility_terms,satisfiable_after_answer=after.satisfiable,
            resolved_before=before,resolved_after=tuple(field for field,value in state.values.items() if value.status == FieldStatus.OBSERVED)))
    return V3EpisodeResult(task_id=original.task_id,policy_name="tr_mla_terminal_aware",initial_state=original.initial_state.public_copy(),
        final_state=state,steps=prefix,solver_calls=2)
