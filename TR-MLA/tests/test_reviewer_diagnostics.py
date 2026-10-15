import importlib
import json
from pathlib import Path
import pytest
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.solver.engine import ConstraintEngine

ROOT = Path(__file__).resolve().parents[1]

def module():
    try:
        return importlib.import_module('paper1_active_req.experiment.reviewer_diagnostics')
    except ModuleNotFoundError:
        pytest.fail('新增诊断模块尚不存在')

def fixture_spec():
    return CompleteSpec.model_validate_json((ROOT/'data/generated/v3/test_complete_specs.jsonl').read_text(encoding='utf-8').splitlines()[0])

def test_masked_parent_state_cannot_leak_target_and_perturbs_only_public_parents():
    m = module()
    spec = fixture_spec()
    state = m.parent_only_state(spec, 'available_width_mm', ('workpiece_width_mm',), None, .05, 20261009)
    assert state.values['available_width_mm'].value is None
    assert state.values['workpiece_mass_kg'].value is None
    assert abs(state.values['workpiece_width_mm'].value/spec.values['workpiece_width_mm'].value-1) == pytest.approx(.05)
    assert spec.values['available_width_mm'].value is not None

def test_six_injections_are_real_unsat_and_have_auditable_changes():
    m = module()
    schema = load_field_schema(ROOT/'configs/fields.yaml')
    engine = ConstraintEngine(schema)
    cases = m.stress_cases(fixture_spec(), schema)
    assert len(cases) == 7
    assert engine.check_satisfiability(cases[0]['state']).satisfiable is True
    for case in cases[1:]:
        assert case['changes']
        result = m.check_with_core(engine, case['state'])
        assert result.satisfiable is False
        assert result.unsat_core
    geometry = next(c for c in cases if c['kind']=='geometry')
    assert geometry['state'].values['available_width_mm'].value == fixture_spec().values['workpiece_width_mm'].value-1

def test_run_exports_per_target_results_and_all_stress_records(tmp_path):
    m = module()
    summary = m.run_diagnostics(ROOT, ROOT/'data/generated/v3', tmp_path, 'test', spec_limit=2)
    assert summary['stress']['perturbed_n']==12
    assert summary['stress']['enabled']['false_accept']==0
    assert summary['stress']['disabled']['false_accept']==12
    assert summary['continuous']['row_count']==60
    assert (tmp_path/'test_continuous.csv').exists()
    assert (tmp_path/'test_stress.jsonl').exists()
    assert summary['input_hashes']
    assert len(summary['parent_availability']) == 15
    assert len(summary['stress']['by_kind']) == 7
    assert summary['stress']['enabled']['false_accept_rate'] == 0
    assert summary['stress']['disabled']['false_accept_rate'] == 1
