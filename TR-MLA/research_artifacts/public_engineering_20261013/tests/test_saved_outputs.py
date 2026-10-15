"""独立从已保存轨迹和协议复核来源、隔离、状态转移与聚合，不调用生成脚本指标函数。"""
import hashlib
import json
from pathlib import Path
from statistics import mean
import pytest
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]

def test_saved_replay_provenance_transitions_and_summary():
    binding=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    protocol=binding['protocol']
    digest=hashlib.sha256(json.dumps(protocol,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest==binding['protocol_hash']
    for path, expected in protocol['input_hashes'].items():
        assert hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()==expected
    cases={c['case_id']:c for c in protocol['cases']}
    snapshots=json.loads((ROOT/'budget_snapshots.json').read_text(encoding='utf-8'))
    episodes=[json.loads(line) for line in (ROOT/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    by_id={e['run_id']:e for e in episodes}
    assert len(by_id)==360 and len(snapshots)==1800
    for e in episodes:
        claims={c['field_id']:c for c in cases[e['case_id']]['claims']}
        state=e['initial_state']; asked=[]
        for f,v in state['values'].items():
            if f in e['hidden_fields']:
                assert v['value'] is None and v['source']=='withheld' and v['status']=='hidden'
            elif f in claims:
                assert v['value']==claims[f]['value']
            else:
                assert v['value'] is None and v['status']=='unknown_to_user'
        for step in e['steps']:
            assert step['public_before']==state
            f=step['field_id']; assert f in e['hidden_fields'] and f not in asked
            assert step['answer']['value']==claims[f]['value']
            assert step['claim_id']==claims[f]['claim_id']
            assert step['scope']==claims[f]['scope'] and step['operator']==claims[f]['operator']
            assert not step['decision'].get('relation_factor_ids',[])
            asked.append(f)
            state=json.loads(json.dumps(state))
            state['values'][f]=step['answer']; state['asked_fields']=asked.copy()
        assert state==e['final_state']
    weights=protocol['schema']
    for r in snapshots:
        ep=by_id[r['run_id']]
        asked=[s['field_id'] for s in ep['steps'][:r['budget']]]
        assert asked==r['asked_fields']
        coverage=sum(weights[f]['criticality'] for f in asked)/sum(weights[f]['criticality'] for f in ep['hidden_fields'])
        assert r['direct_weighted_coverage']==pytest.approx(coverage)
        assert r['answer_cost']==pytest.approx(sum(weights[f]['answer_cost'] for f in asked))
    outcome=json.loads((ROOT/'outcome.json').read_text(encoding='utf-8'))
    for row in outcome['summary']:
        group=[r for r in snapshots if r['policy']==row['policy'] and r['budget']==row['budget']]
        assert len(group)==90
        assert row['direct_weighted_coverage_mean']==pytest.approx(mean(r['direct_weighted_coverage'] for r in group))
        assert row['answer_cost_mean']==pytest.approx(mean(r['answer_cost'] for r in group))
    assert outcome['project_count']==1 and outcome['subsystem_count']==2
    assert outcome['complete_source_relation_chains']==0

def test_all_budget_prefix_decisions_use_only_available_state():
    import importlib.util
    spec=importlib.util.spec_from_file_location('replay_impl',PROJECT/'scripts/run_public_engineering_20261013.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    schema=m.load_field_schema(PROJECT/'configs/fields.yaml'); config=m.load_v3_relation_config(PROJECT/'configs/v3_relations.yaml')
    episodes=[json.loads(line) for line in (ROOT/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    for seed in m.SEEDS:
        path=PROJECT/f'research_artifacts/reviewer_supplement_20261009/datasets/seed-{seed}/train_complete_specs.jsonl'
        train=tuple(m.CompleteSpec.model_validate_json(line) for line in path.read_text(encoding='utf-8').splitlines())
        model=m.TypedRelationalModel.fit(train,schema,config)
        for ep in (e for e in episodes if e['seed']==seed):
            initial=m.RequirementState.model_validate(ep['initial_state']);engine=m.DomainOnlyEngine(schema)
            policy=m.make_policy(ep['policy'],initial,model,engine,train,schema,config)
            for budget in range(1,5):
                state=initial.public_copy()
                for i,step in enumerate(ep['steps'][:budget]):
                    decision=policy.select(state.public_copy(),engine.check_satisfiability(state),m.PolicyContext(field_schema=schema,budget=budget,step_index=i))
                    assert decision.field_id==step['field_id']
                    state=state.apply_answer(step['field_id'],m.FieldValue.model_validate(step['answer']))
