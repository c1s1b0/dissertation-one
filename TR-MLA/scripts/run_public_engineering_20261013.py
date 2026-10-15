"""公开采购需求参数回放：冻结核心策略，实验侧语义/可回答性/约束层适配。"""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
from statistics import mean
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT/'research_artifacts/public_engineering_20261013'
sys.path.insert(0,str(PROJECT/'src'))
from paper1_active_req.domain.models import FieldValue, FieldStatus, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.policies.baselines import CriticalityGreedyPolicy
from paper1_active_req.policies.marginal_lookahead import TRMLAPolicy, TypedRelationalMyopicPolicy, MarginalLookaheadNoRelationsPolicy

SEEDS = (20261011,20261013,20261017)
POLICIES = ('tr_mla','typed_relational_myopic','criticality_greedy','marginal_lookahead_no_relations')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def write_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def evidence(start,end,quote):
    return dict(start=start,end=end,quote=quote)


def case_definitions():
    common = [
        dict(field_id='workpiece_length_mm',value=600.0,unit='mm',operator='=',semantic_role='要求箱体长度包络的上下界相同',scope='立体库处理的单箱',evidence=[evidence(99,99,'最大尺寸：600×500×350mm，最小尺寸：600×400×170mm')],inheritance='二、立体库基本要求向处理料箱的本子系统继承'),
        dict(field_id='workpiece_mass_kg',value=30.0,unit='kg',operator='<=',semantic_role='单箱质量需求上限参数，不是实测质量',scope='立体库处理的单箱',evidence=[evidence(100,100,'料箱重量：≤30Kg')],inheritance='二、立体库基本要求向处理料箱的本子系统继承'),
    ]
    shuttle = dict(project_id='postal_qingdao_20260708',case_id='shuttle_travel',label='穿梭车行走子系统',claims=json.loads(json.dumps(common))+[
        dict(field_id='load_capacity_kg',value=35.0,unit='kg',operator='rated',semantic_role='额定载货质量要求',scope='穿梭车额定载货',evidence=[evidence(367,368,'额定载货质量\n35kg')],inheritance=None),
        dict(field_id='positioning_accuracy_mm',value=3.0,unit='mm',operator='abs_error<=',semantic_role='绝对定位误差容限参数，不是实测定位误差',scope='穿梭车行走轴',evidence=[evidence(373,381,'定位精度\n±3mm')],inheritance=None),
    ])
    cargo = dict(project_id='postal_qingdao_20260708',case_id='cargo_lift',label='货物提升机子系统',claims=json.loads(json.dumps(common))+[
        dict(field_id='load_capacity_kg',value=70.0,unit='kg',operator='rated',semantic_role='双箱位合计额定荷载要求',scope='双箱位总额定载荷',evidence=[evidence(450,451,'额定荷载能力\n70kg（双箱位）'),evidence(131,131,'荷载：双箱位，2*35kg')],inheritance=None),
        dict(field_id='positioning_accuracy_mm',value=2.0,unit='mm',operator='abs_error<=',semantic_role='绝对定位误差容限参数，不是实测定位误差',scope='货物提升机升降轴',evidence=[evidence(456,457,'定位精度\n±2mm')],inheritance=None),
    ])
    for case in (shuttle,cargo):
        for claim in case['claims']:
            claim['claim_id'] = case['case_id']+':'+claim['field_id']
    return [shuttle,cargo]


class DomainOnlyEngine(ConstraintEngine):
    """在基础求解器创建前清除合成规则；从未加载其约束。"""
    def _build_base_satisfiability_solver(self):
        self.rules = []
        return super()._build_base_satisfiability_solver()


def make_initial(case, hidden, schema):
    claims = {x['field_id']:x for x in case['claims']}
    values = {}
    for f in schema:
        if f not in claims:
            values[f] = FieldValue(value=None,status=FieldStatus.UNKNOWN_TO_USER,source='not_available_for_scalar_replay')
        elif f in hidden:
            values[f] = FieldValue(value=None,status=FieldStatus.HIDDEN,source='withheld')
        else:
            values[f] = FieldValue(value=claims[f]['value'],status=FieldStatus.OBSERVED,source='public_requirement_parameter')
    return RequirementState(values=values)


def masks(case):
    fields = sorted(c['field_id'] for c in case['claims'])
    return [list(comb) for n in range(1,len(fields)+1) for comb in itertools.combinations(fields,n)]


def input_paths():
    paths = {str(p.relative_to(PROJECT)):p for p in sorted((PROJECT/'src/paper1_active_req').rglob('*.py'))}
    paths.update({str(p.relative_to(PROJECT)):p for p in [PROJECT/'configs/fields.yaml',PROJECT/'configs/v3_relations.yaml',Path(__file__),ROOT/'sources/postal.txt',ROOT/'sources/postal.html',ROOT/'sources/manifest.json',ROOT/'tests/test_public_replay.py'] if p.exists()})
    for seed in SEEDS:
        p=PROJECT/f'research_artifacts/reviewer_supplement_20261009/datasets/seed-{seed}/train_complete_specs.jsonl'
        paths[str(p.relative_to(PROJECT))]=p
    return paths


def freeze():
    schema=load_field_schema(PROJECT/'configs/fields.yaml')
    cases=case_definitions()
    lines=(ROOT/'sources/postal.txt').read_text(encoding='utf-8-sig').splitlines()
    for case in cases:
        for c in case['claims']:
            for e in c['evidence']:
                assert e['quote'] in '\n'.join(lines[e['start']-1:e['end']])
            assert schema[c['field_id']].lower <= c['value'] <= schema[c['field_id']].upper
    protocol=dict(version=1,label='公开需求参数回放／来源约束的迁移诊断',source_url='https://www.chinapost.com.cn/cn/report/2607/2069-1.htm',source_date='2026-07-08',project_count=1,subsystem_count=2,cases=cases,training_seeds=list(SEEDS),policies=list(POLICIES),lambda_cost=0.0,lambda_origin='沿用reviewer_supplement_20261009的primary_lambda，无本来源调参',budgets=list(range(5)),mask_plan={c['case_id']:masks(c) for c in cases},schema={f:d.model_dump(mode='json') for f,d in schema.items()},weights_costs='全部沿用benchmark固定设定，非真实工程师权重或访谈成本',engine='仅原schema域约束，18条benchmark工程规则均不加载；这些范围是计算支持域而非源文档工程要求',missing_behavior='非可用标量一律unknown_to_user；不从合成数据或跨设备补值；仅4个需求参数可询问',prefix_plan='每条完整序列选到min(4,hidden_count)，输出预算0..4的前缀；原四selector不读取context.budget；超出hidden_count保持终态',semantic_adaptation='value表示原文要求的标量参数或阈值，而不是实际工况值；比较符和作用域保留于外部来源记录；冻结原合成prior仅作探索性迁移',metrics=['direct_weighted_coverage: 初始隐藏字段中已询问且来源回答正确的criticality权重比例','answer_cost: 所询字段原benchmark成本之和','selected_sequence: 字段选择顺序','used_relation_factor_ids: 选择分数实际用到的关系因子'],relation_coverage='0条完整且适用于可评分字段的来源父子链；不得把此回放解释为真实关系收益验证',excluded=[dict(fields=['workpiece_width_mm','workpiece_height_mm'],reason='原文宽[400,500]、高[170,350]为区间；无原策略区间支持，禁止取中点或端点充当值'),dict(fields=['conveyor_speed_mm_s','travel_distance_mm','target_throughput_per_hour'],reason='升降或shuttle行走速度非输送机；最大高度非实际行程；系统或最大效率非单机目标需求'),dict(fields=['repeatability_mm','safety_factor','workpiece_fragile','available_length_mm','available_width_mm','available_height_mm','side_clearance_mm','cycle_time_s','transfer_time_s'],reason='来源缺少相同作用域和含义的可用标量答案'),dict(fields=['interfaces','power_supply'],reason='驱动带非货物接口，PLC电源非整机电源；不推断')],inference='报告描述性枚举均值；不计算显著性或将mask/seed当独立项目；不并入合成测试汇总',input_hashes={k:sha(v) for k,v in input_paths().items()})
    digest=hashlib.sha256(canonical(protocol).encode()).hexdigest()
    payload=dict(protocol_hash=digest,protocol=protocol)
    target=ROOT/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text(encoding='utf-8'))==payload,'协议或输入已变化，禁止沿用结果'
    else:
        write_json(target,payload)
    return payload


def make_policy(name,state,model,engine,train,schema,config):
    if name=='criticality_greedy':
        return CriticalityGreedyPolicy()
    if name=='marginal_lookahead_no_relations':
        return MarginalLookaheadNoRelationsPolicy.from_training_specs(initial_state=state,training_specs=train,schema=schema,relation_config=config,engine=engine,lambda_cost=0.0)
    cls=TRMLAPolicy if name=='tr_mla' else TypedRelationalMyopicPolicy
    return cls(initial_state=state,model=model,engine=engine,lambda_cost=0.0)


def validate_metrics(rows,schema):
    for r in rows:
        asked=r['asked_fields']
        assert len(asked)==len(set(asked))
        assert set(asked)<=set(r['hidden_fields'])
        assert len(asked)<=r['budget']
        denominator=sum(schema[f].criticality for f in r['hidden_fields'])
        expected=sum(schema[f].criticality for f in asked)/denominator
        assert abs(expected-r['direct_weighted_coverage'])<1e-12
        assert abs(sum(schema[f].answer_cost for f in asked)-r['answer_cost'])<1e-12


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:canonical(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)


def run():
    frozen=freeze()
    schema=load_field_schema(PROJECT/'configs/fields.yaml')
    config=load_v3_relation_config(PROJECT/'configs/v3_relations.yaml')
    cases=frozen['protocol']['cases']
    snapshots=[]; episodes=[]; steps=[]
    for seed in SEEDS:
        path=PROJECT/f'research_artifacts/reviewer_supplement_20261009/datasets/seed-{seed}/train_complete_specs.jsonl'
        train=tuple(CompleteSpec.model_validate_json(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())
        assert len(train)==180 and all(x.split=='train' for x in train)
        model=TypedRelationalModel.fit(train,schema,config)
        for case in cases:
            claims={c['field_id']:c for c in case['claims']}
            for mask_index,hidden in enumerate(masks(case)):
                for name in POLICIES:
                    initial=make_initial(case,hidden,schema)
                    state=initial.public_copy()
                    engine=DomainOnlyEngine(schema)
                    policy=make_policy(name,initial,model,engine,train,schema,config)
                    run_id=f"{seed}:{case['case_id']}:{mask_index:02d}:{name}"
                    episode=dict(run_id=run_id,seed=seed,case_id=case['case_id'],mask_id=str(mask_index),policy=name,hidden_fields=hidden,initial_state=initial.model_dump(mode='json'),steps=[])
                    for k in range(min(4,len(hidden))):
                        public=state.public_copy()
                        assert all(public.values[f].value is None for f in hidden if f not in state.asked_fields)
                        solver_result=engine.check_satisfiability(public)
                        assert solver_result.satisfiable is True
                        decision=policy.select(public,solver_result,PolicyContext(field_schema=schema,budget=4,step_index=k))
                        if decision.field_id is None:
                            break
                        f=decision.field_id
                        assert f in hidden and f in state.queryable_field_ids()
                        claim=claims[f]
                        answer=FieldValue(value=claim['value'],status=FieldStatus.OBSERVED,source='public_requirement_parameter')
                        state=state.apply_answer(f,answer)
                        step=dict(run_id=run_id,step_index=k+1,field_id=f,answer=answer.model_dump(mode='json'),claim_id=claim['claim_id'],operator=claim['operator'],semantic_role=claim['semantic_role'],scope=claim['scope'],decision=decision.model_dump(mode='json'),public_before=public.model_dump(mode='json'))
                        episode['steps'].append(step); steps.append(step)
                    episode['final_state']=state.model_dump(mode='json')
                    episodes.append(episode)
                    for budget in range(5):
                        prefix=episode['steps'][:budget]
                        asked=[x['field_id'] for x in prefix]
                        snapshots.append(dict(protocol_hash=frozen['protocol_hash'],run_id=run_id,seed=seed,case_id=case['case_id'],mask_id=str(mask_index),policy=name,budget=budget,hidden_count=len(hidden),hidden_fields=hidden,asked_fields=asked,direct_weighted_coverage=sum(schema[f].criticality for f in asked)/sum(schema[f].criticality for f in hidden),answer_cost=sum(schema[f].answer_cost for f in asked),used_relation_factor_ids=sorted({f for st in prefix for f in st['decision'].get('relation_factor_ids',[])})))
        print(f'完成训练seed {seed}：累计{len(episodes)}条回放轨迹',flush=True)
    validate_metrics(snapshots,schema)
    assert len(episodes)==360 and len(snapshots)==1800
    assert not any(row['used_relation_factor_ids'] for row in snapshots)
    # 保存记录再从磁盘重算，避免仅相信内存汇总。
    (ROOT/'episodes.jsonl').write_text(''.join(canonical(x)+'\n' for x in episodes),encoding='utf-8')
    (ROOT/'steps.jsonl').write_text(''.join(canonical(x)+'\n' for x in steps),encoding='utf-8')
    write_json(ROOT/'budget_snapshots.json',snapshots)
    save_csv(ROOT/'budget_snapshots.csv',snapshots)
    save_csv(ROOT/'steps.csv',steps)
    loaded=json.loads((ROOT/'budget_snapshots.json').read_text(encoding='utf-8'))
    validate_metrics(loaded,schema)
    summary=[]
    for name in POLICIES:
        for budget in range(5):
            rows=[r for r in loaded if r['policy']==name and r['budget']==budget]
            summary.append(dict(policy=name,budget=budget,direct_weighted_coverage_mean=mean(r['direct_weighted_coverage'] for r in rows),answer_cost_mean=mean(r['answer_cost'] for r in rows),query_count_mean=mean(len(r['asked_fields']) for r in rows),descriptive_replay_count=len(rows),project_count=1))
    save_csv(ROOT/'summary.csv',summary)
    comparisons=[]
    for other in POLICIES[1:]:
        a={(e['seed'],e['case_id'],e['mask_id']):[s['field_id'] for s in e['steps']] for e in episodes if e['policy']=='tr_mla'}
        b={(e['seed'],e['case_id'],e['mask_id']):[s['field_id'] for s in e['steps']] for e in episodes if e['policy']==other}
        comparisons.append(dict(other=other,sequence_identical=sum(a[k]==b[k] for k in a),descriptive_pairs=len(a)))
    outcome=dict(protocol_hash=frozen['protocol_hash'],completed_utc=datetime.now(timezone.utc).isoformat(),project_count=1,subsystem_count=2,episode_count=len(episodes),snapshot_count=len(snapshots),step_count=len(steps),complete_source_relation_chains=0,relation_factor_uses=0,summary=summary,sequence_comparisons=comparisons,python=platform.python_version())
    write_json(ROOT/'outcome.json',outcome)
    freeze() # 再核验所有冻结输入及核心源码哈希。
    write_json(ROOT/'verification.json',dict(protocol_hash=frozen['protocol_hash'],source_quotes_verified=True,scalar_values_in_original_domains=True,unknown_fields_excluded=True,hidden_values_removed=True,domain_only_rules_empty=True,metrics_recomputed_from_disk=True,core_input_hashes_unchanged=True,output_hashes={p.name:sha(p) for p in [ROOT/'episodes.jsonl',ROOT/'steps.jsonl',ROOT/'budget_snapshots.json',ROOT/'budget_snapshots.csv',ROOT/'steps.csv',ROOT/'summary.csv',ROOT/'outcome.json']}))
    print(canonical(outcome),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args()
    if args.freeze_only:
        print(canonical(freeze()),flush=True)
    else:
        run()

