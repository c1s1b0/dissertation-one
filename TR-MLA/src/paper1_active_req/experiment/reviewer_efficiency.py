"""审稿后独立效率测量；不改变冻结策略与求解器语义。"""
from contextlib import contextmanager
from time import perf_counter

import z3

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.solver.engine import ConstraintEngine


class InstrumentedEngine(ConstraintEngine):
    """串行专用计数子类；实际check调用包含unknown，不把缓存命中计为SAT。"""

    def __init__(self, *args, **kwargs):
        self.sat_calls = 0
        self.branch_requests = 0
        self.unknown_checks = 0
        self._instrumenting = False
        super().__init__(*args, **kwargs)

    @contextmanager
    def _count_z3(self):
        if self._instrumenting:
            yield
            return
        original = z3.Solver.check
        owner = self

        def counted(solver, *args, **kwargs):
            owner.sat_calls += 1
            result = original(solver, *args, **kwargs)
            owner.unknown_checks += int(result == z3.unknown)
            return result

        self._instrumenting = True
        z3.Solver.check = counted
        try:
            yield
        finally:
            z3.Solver.check = original
            self._instrumenting = False

    def check(self, state):
        with self._count_z3():
            return super().check(state)

    def check_satisfiability(self, state):
        self.branch_requests += 1
        with self._count_z3():
            return super().check_satisfiability(state)


def scale_mask(state, schema, candidate_count):
    """固定24字段、全部隐藏，再按字典序开放n个字段；绝不增加公开真值。"""
    ids = sorted(k for k, v in schema.items() if v.queryable)
    if not 0 <= candidate_count <= len(ids):
        raise ValueError("candidate_count outside queryable schema")
    selected = set(ids[:candidate_count])
    limited = {k: v.model_copy(update={"queryable": k in selected}) for k, v in schema.items()}
    masked = RequirementState(values={k: FieldValue(value=None, status=FieldStatus.HIDDEN,
                                                     source="artificial-scale-mask") for k in schema})
    return masked, limited


def measure_decision(policy, engine, state, boundary, context, *, clock=perf_counter):
    """准备与模型拟合由调用者完成；仅select位于计时窗口。"""
    sat_before, branch_before, unknown_before = engine.sat_calls, engine.branch_requests, engine.unknown_checks
    start = clock()
    decision = policy.select(state, boundary, context)
    elapsed = clock() - start
    # 查询放在计时之后，避免提前调用posterior温热策略缓存。
    if hasattr(policy, "scorer"):
        eligible = policy.scorer._candidate_ids(state, context)
    else:
        eligible = [k for k in state.queryable_field_ids() if context.field_schema[k].queryable]
    return {"decision_seconds": elapsed, "actual_eligible": len(eligible),
            "selected_field": decision.field_id,
            "sat_calls": engine.sat_calls - sat_before,
            "counterfactual_branch_requests": engine.branch_requests - branch_before,
            "unknown_checks": engine.unknown_checks - unknown_before,
            "selected_action_branch_count": decision.utility_terms.get("counterfactual_branch_count", 0)}
