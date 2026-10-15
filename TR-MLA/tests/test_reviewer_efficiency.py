"""效率协议的行为测试；新增模块缺失时首先失败。"""
import importlib
import importlib.util

from pathlib import Path
import pytest

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.policies.baselines import CriticalityGreedyPolicy


def module():
    name = "paper1_active_req.experiment.reviewer_efficiency"
    assert importlib.util.find_spec(name) is not None, "效率协议模块尚未实现"
    return importlib.import_module(name)


def test_scale_mask_never_discloses_hidden_truth_and_keeps_24_fields():
    m = module()
    schema = load_field_schema(Path("configs/fields.yaml"))
    state = RequirementState(values={k: FieldValue(value=123, status=FieldStatus.OBSERVED) for k in schema})
    masked, limited = m.scale_mask(state, schema, 4)
    assert len(limited) == len(masked.values) == 24
    assert sum(v.queryable for v in limited.values()) == 4
    assert all(v.value is None and v.status == FieldStatus.HIDDEN for v in masked.values.values())
    assert all(v.value == 123 for v in state.values.values())
    with pytest.raises(ValueError):
        m.scale_mask(state, schema, 25)


def test_sat_instrumentation_counts_requests_separately_from_actual_z3_calls():
    m = module()
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = m.InstrumentedEngine(schema)
    state = RequirementState(values={k: FieldValue(value=None, status=FieldStatus.HIDDEN) for k in schema})
    first = engine.check_satisfiability(state)
    second = engine.check_satisfiability(state)
    assert first.satisfiable == second.satisfiable
    assert engine.branch_requests == 2
    assert engine.sat_calls == 1


def test_pure_decision_timer_excludes_boundary_and_reports_real_candidate_count():
    m = module()
    schema = load_field_schema(Path("configs/fields.yaml"))
    initial = RequirementState(values={k: FieldValue(value=None, status=FieldStatus.HIDDEN) for k in schema})
    state, limited = m.scale_mask(initial, schema, 2)
    engine = m.InstrumentedEngine(limited)
    boundary = engine.check_satisfiability(state)
    ticks = iter((10.0, 10.25))
    row = m.measure_decision(CriticalityGreedyPolicy(), engine, state, boundary,
                             PolicyContext(field_schema=limited, budget=8, step_index=0),
                             clock=lambda: next(ticks))
    assert row["decision_seconds"] == 0.25
    assert row["actual_eligible"] == 2
    assert row["sat_calls"] == row["counterfactual_branch_requests"] == 0
    assert row["selected_field"] in state.queryable_field_ids()

