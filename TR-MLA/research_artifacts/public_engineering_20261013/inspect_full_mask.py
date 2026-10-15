import importlib.util,json
from pathlib import Path
p=Path('E:/多智能体/paper1_experiment')
s=importlib.util.spec_from_file_location('r',p/'scripts/run_public_engineering_20261013.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
schema=m.load_field_schema(p/'configs/fields.yaml');cfg=m.load_v3_relation_config(p/'configs/v3_relations.yaml')
train=tuple(m.CompleteSpec.model_validate_json(l) for l in (p/'research_artifacts/reviewer_supplement_20261009/datasets/seed-20261011/train_complete_specs.jsonl').read_text().splitlines())
model=m.TypedRelationalModel.fit(train,schema,cfg)
case=m.case_definitions()[0];hidden=[c['field_id'] for c in case['claims']];initial=m.make_initial(case,hidden,schema)
result=[]
for name in m.POLICIES:
 state=initial.public_copy();engine=m.DomainOnlyEngine(schema);policy=m.make_policy(name,initial,model,engine,train,schema,cfg);seq=[]
 for i in range(4):
  ctx=m.PolicyContext(field_schema=schema,budget=4,step_index=i)
  scores=[]
  if hasattr(policy,'scorer'):
   for f in state.queryable_field_ids():
    a=policy.scorer.score_action(state,f,ctx)
    scores.append(dict(field_id=f,total=a.total,first_gain=a.first_step_gain,second_gain=a.second_step_gain,cost=a.normalized_answer_cost))
  dec=policy.select(state.public_copy(),engine.check_satisfiability(state),ctx)
  # 显式检查预算字段不改变同一公开状态下的选择，前缀可对应独立预算运行。
  for budget in range(i+1,5):
   check=policy.select(state.public_copy(),engine.check_satisfiability(state),m.PolicyContext(field_schema=schema,budget=budget,step_index=i))
   assert dec.field_id==check.field_id
  f=dec.field_id; c=next(c for c in case['claims'] if c['field_id']==f)
  state=state.apply_answer(f,m.FieldValue(value=c['value'],status=m.FieldStatus.OBSERVED,source='public_requirement_parameter'))
  seq.append(dict(step=i+1,selected=f,value=c['value'],operator=c['operator'],weighted_coverage=sum(schema[q].criticality for q in state.asked_fields)/3.8,candidate_scores=scores))
 result.append(dict(policy=name,steps=seq))
m.write_json(m.ROOT/'full_mask_example.json',dict(seed=20261011,case_id=case['case_id'],denominator=3.8,budget_prefix_invariance_verified=True,policies=result))
print(json.dumps(result,ensure_ascii=False,indent=2))
