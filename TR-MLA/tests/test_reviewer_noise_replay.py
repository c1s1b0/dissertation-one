import importlib
import json
from pathlib import Path
import pytest
from paper1_active_req.domain.models import FieldValue, FieldStatus, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.solver.engine import ConstraintEngine

ROOT = Path(__file__).resolve().parents[1]

def module():
    try:
        return importlib.import_module('paper1_active_req.experiment.reviewer_noise_replay')
    except ModuleNotFoundError:
        pytest.fail('缺少独立固定序列含噪重放模块')

def test_noise_uses_shared_spec_field_draws_and_preserves_original():
    m = module()
    schema = load_field_schema(ROOT/'configs/fields.yaml')
    answer = {'value': 100., 'status':'observed', 'source':'oracle'}
    first = m.noisy_answer(answer, schema['working_height_mm'], 'spec-a', 1., .1)
    second = m.noisy_answer(answer, schema['working_height_mm'], 'spec-a', 1., .1)
    assert first == second
    assert abs(first['answer']['value']/100-1) == pytest.approx(.1)
    assert first['out_of_domain']
    assert answer['value']==100.
    clean = m.noisy_answer(answer, schema['working_height_mm'], 'spec-a', 0., .1)
    assert clean['answer']==answer
    assert clean['contaminated'] is False

def test_guard_rejects_bad_candidate_preserves_state_and_consumes_one_step():
    m = module()
    schema = load_field_schema(ROOT/'configs/fields.yaml')
    spec = json.loads((ROOT/'data/generated/v3/test_complete_specs.jsonl').read_text(encoding='utf-8').splitlines()[0])
    state = RequirementState.model_validate({'values':spec['values']})
    before = state.model_dump()
    mutated = {'value':schema['working_height_mm'].upper+1, 'status':'observed', 'source':'noise'}
    final, event = m.apply_candidate(state, 'working_height_mm', mutated, ConstraintEngine(schema), True)
    assert final.model_dump()==before
    assert event['accepted'] is False
    assert event['budget_consumed']==1
    assert event['candidate_sat'] is False
    final, event = m.apply_candidate(state, 'working_height_mm', mutated, ConstraintEngine(schema), False)
    assert event['accepted'] is True
    assert final.values['working_height_mm'].value==mutated['value']

def test_replay_clean_matches_common_metrics_and_noise_guards_share_answers(tmp_path):
    m = module()
    result = m.run_noise_replay(ROOT, tmp_path, episode_limit=4)
    assert result['original_episode_count']==4
    assert result['replay_count']==40
    records = [json.loads(x) for x in (tmp_path/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    for i in range(0,len(records),2):
        off,on=records[i:i+2]
        assert off['contaminated_answer_count']==on['contaminated_answer_count']
        assert [x['new_answer'] for x in off['steps']]==[x['new_answer'] for x in on['steps']]
        assert off['actual_question_count']==on['actual_question_count']
        if off['error_probability']==0:
            assert off['metrics']==on['metrics']
            assert off['clean_metric_matches_original_reviewer']
