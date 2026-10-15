import importlib.util
from pathlib import Path
import pytest
P = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('public_replay', P/'scripts/run_public_engineering_20261013.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_hidden_and_unknown_do_not_leak():
    schema = m.load_field_schema(m.PROJECT/'configs/fields.yaml')
    case = m.case_definitions()[0]
    hidden = [x['field_id'] for x in case['claims']]
    state = m.make_initial(case, hidden, schema)
    assert set(state.queryable_field_ids()) == set(hidden)
    assert all(state.values[f].value is None for f in hidden)
    assert all(v.value is None for f,v in state.values.items() if f not in hidden)
    assert all(v.source == 'withheld' for f,v in state.values.items() if f in hidden)

def test_domain_engine_does_not_inherit_synthetic_safety_rules():
    schema = m.load_field_schema(m.PROJECT/'configs/fields.yaml')
    engine = m.DomainOnlyEngine(schema)
    state = m.make_initial(m.case_definitions()[0], [], schema)
    state.values['safety_factor'] = m.FieldValue(value=1.2,status=m.FieldStatus.OBSERVED)
    assert engine.rules == []
    assert engine.check_satisfiability(state).satisfiable is True

def test_provenance_exact_and_no_interval_midpoints():
    lines = (m.ROOT/'sources/postal.txt').read_text(encoding='utf-8-sig').splitlines()
    for case in m.case_definitions():
        for claim in case['claims']:
            for evidence in claim['evidence']:
                assert evidence['quote'] in '\n'.join(lines[evidence['start']-1:evidence['end']])
            assert claim['field_id'] not in {'workpiece_width_mm','workpiece_height_mm','conveyor_speed_mm_s','target_throughput_per_hour','travel_distance_mm'}
    assert m.case_definitions()[1]['claims'][2]['scope'] == '双箱位总额定载荷'

def test_budget_zero_and_recomputed_metrics():
    schema = m.load_field_schema(m.PROJECT/'configs/fields.yaml')
    hidden = ['workpiece_mass_kg', 'positioning_accuracy_mm']
    rows = [dict(case_id='x',seed=1,mask_id='m',policy='p',budget=0,hidden_fields=hidden,asked_fields=[],direct_weighted_coverage=0.0,answer_cost=0.0),dict(case_id='x',seed=1,mask_id='m',policy='p',budget=1,hidden_fields=hidden,asked_fields=['workpiece_mass_kg'],direct_weighted_coverage=0.5,answer_cost=0.2)]
    m.validate_metrics(rows,schema)
    rows[1]['direct_weighted_coverage']=1.0
    with pytest.raises(AssertionError):
        m.validate_metrics(rows,schema)


def test_claim_ids_bound_to_subsystem():
    for case in m.case_definitions():
        assert all(c['claim_id'].startswith(case['case_id']+':') for c in case['claims'])
