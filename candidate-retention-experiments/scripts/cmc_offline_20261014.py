"""CMC 第二轮离线补强：不读取环境配置，不实例化客户端，不调用模型。

Support 只读取目录属性、单变量边界和二元关系。可行解仅用于事后核验。
默认读取随包脱敏审计；--export-audit 从本地旧日志导出固定白名单。
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import reconstruct_paper2_20261013 as old
from paper2_dynamic_agents.experiment.solver import _ports_compatible

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research_artifacts/cmc-offline-20261014'
AUDIT=ROOT/'research_artifacts/cmc-audit-input-20261014'
OLD=ROOT/'research_artifacts/reconstruction-20261013'
STRATA=[('A','configuration'),('A','repair'),('B','configuration'),('B','repair')]

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def relation_edges(task):
    numeric=([('drive','conveyor'),('positioner','conveyor'),('transmission','drive')] if task.scene=='A'
             else [('motor','gearbox'),('motor','coupling'),('motor','shaft_support')])
    adjacent=list(zip(task.required_roles,task.required_roles[1:]))
    return sorted({tuple(sorted(e)) for e in numeric+adjacent})

def pair_compatible(task,a,b):
    c={a.role:a,b.role:b};p={r:x.parameters for r,x in c.items()}
    if task.scene=='A':
        if {'drive','conveyor'}<=c.keys() and p['drive']['capacity_kg']<p['conveyor']['load_kg']:return False
        if {'positioner','conveyor'}<=c.keys() and p['positioner']['payload_kg']<p['conveyor']['load_kg']:return False
        if {'transmission','drive'}<=c.keys() and p['transmission']['speed_rpm']>2*p['drive']['speed_rpm']:return False
    elif task.scene=='B':
        if {'motor','gearbox'}<=c.keys() and p['gearbox']['input_speed_rpm']>p['motor']['speed_rpm']:return False
        if {'motor','coupling'}<=c.keys() and (p['coupling']['torque_nm']<p['motor']['torque_nm'] or p['coupling']['bore_mm']<p['motor']['shaft_diameter_mm']):return False
        if {'motor','shaft_support'}<=c.keys() and p['shaft_support']['bore_mm']<p['motor']['shaft_diameter_mm']:return False
    else:raise ValueError(task.scene)
    for l,r in zip(task.required_roles,task.required_roles[1:]):
        if {l,r}==set(c) and not _ports_compatible(c[l],c[r]):return False
    return True

def support_pool(task,items,p_ids,edges,compatible):
    by={c.template_id:c for c in items};initial={v for vs in task.candidate_ids.values() for v in vs}
    target=set(p_ids)
    if len(target)!=len(p_ids) or not initial<=target or not target<=by.keys():raise ValueError('invalid target pool')
    unary={c.template_id:old.unary_score(task,c) for c in items}
    domains={r:[c for c in items if c.role==r and unary[c.template_id][0]==0] for r in task.required_roles}
    neighbors={r:set() for r in task.required_roles}
    for a,b in edges:neighbors[a].add(b);neighbors[b].add(a)
    score={};checks=0
    for c in items:
        total=0.
        for r in sorted(neighbors[c.role]):
            ds=domains[r]
            total+=sum(compatible(c,d) for d in ds)/len(ds) if ds else 0.
            checks+=len(ds)
        score[c.template_id]=total
    result=set(initial)
    for r in task.required_roles:
        q=sum(by[i].role==r for i in target)-len(task.candidate_ids[r])
        if q<0:raise ValueError('negative role quota')
        options=[c for c in items if c.role==r and c.template_id not in initial]
        options.sort(key=lambda c:(unary[c.template_id][0]>0,-score[c.template_id],*unary[c.template_id],c.template_id))
        result.update(c.template_id for c in options[:q])
        assert sum(by[i].role==r for i in result)==sum(by[i].role==r for i in target)
    return tuple(c.template_id for c in items if c.template_id in result),{'pair_checks':checks,'support_scores':score}

def summarize_audit(rows):
    keys={(r['task_id'],r['repeat']) for r in rows}
    retried={(r['task_id'],r['repeat']) for r in rows if r.get('retry_index',0)>0 or r.get('task_attempt',1)>1}
    def valid_usage(r):
        u=r.get('usage') or {};v=u.get('total_tokens')
        return isinstance(v,(int,float)) and not isinstance(v,bool) and v>0
    covered=[r for r in rows if valid_usage(r)]
    accepted=sum(r['status']=='success' and r.get('http_status')==200 for r in rows)
    invalid=sum(r.get('error_type') in ('ValidationError','ContractCompletionError','JsonContractError') for r in rows)
    return dict(request_attempts=len(rows),task_repeats=len(keys),retried_task_repeats=len(retried),
      validation_error_records=invalid,validation_outcomes=accepted+invalid,
      audit_acceptance_rate=accepted/(accepted+invalid) if accepted+invalid else None,
      internal_retry_attempts=sum(r.get('retry_index',0)>0 for r in rows),
      rerun_task_repeats=len({(r['task_id'],r['repeat']) for r in rows if r.get('task_attempt',1)>1}),
      failed_attempts=sum(r['status']!='success' for r in rows),
      non200_or_missing_attempts=sum(r.get('http_status')!=200 for r in rows),
      http200_attempts=sum(r.get('http_status')==200 for r in rows),
      accepted_proposals=sum(r['status']=='success' and r.get('http_status')==200 for r in rows),
      rejected_http200=sum(r['status']!='success' and r.get('http_status')==200 for r in rows),
      positive_usage_attempts=len(covered),
      zero_usage_attempts=sum((r.get('usage') or {}).get('total_tokens')==0 for r in rows),
      missing_usage_attempts=sum((r.get('usage') or {}).get('total_tokens') is None for r in rows),
      reported_total_tokens=sum(r['usage']['total_tokens'] for r in covered),
      reported_prompt_tokens=sum(r['usage'].get('prompt_tokens',0) for r in covered),
      reported_completion_tokens=sum(r['usage'].get('completion_tokens',0) for r in covered),
      request_latency_ms_quantiles=[float(v) for v in np.quantile([r['latency_ms'] for r in rows],[.5,.9,.95])],
      error_counts=dict(Counter(r.get('error_type') or 'none' for r in rows)),
      total_billed_tokens=None,monetary_cost=None)

def export_audit():
    path=ROOT/'research_artifacts/formal_llm_runs/main-v5-postaudit-test-20260817/llm_audit.jsonl'
    fields=['task_id','repeat','stage','task_attempt','attempt','retry_index','status','http_status','error_type','latency_ms','usage']
    rows=[]
    for line,r in enumerate(old.jsonl(path),1):
        if r['method']=='P':
            v={f:r.get(f) for f in fields}
            if v['usage'] is not None:v['usage']={k:v['usage'].get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')}
            rows.append(dict(source_row=line,**v))
    dump(AUDIT/'p_attempts_sanitized.json',rows)
    dump(AUDIT/'provenance.json',dict(source_relative_path=path.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       export_fields=['source_row']+fields,policy='Whitelist only; no endpoint, request ID, timestamps, provider identity, prompts or response contents. Historical provider token counters are not billing evidence.'))

def verify_pair_factorization(task,items,edges):
    # 独立调用原裁决器核对分解，覆盖所有二元候选对的不同背景组合；不是排名输入。
    byrole={r:[c for c in items if c.role==r] for r in task.required_roles};by={c.template_id:c for c in items}
    for offset in (0,7,13):
        for a,b in itertools.combinations(task.required_roles,2):
            for x,y in itertools.product(byrole[a],byrole[b]):
                chosen={r:byrole[r][offset%len(byrole[r])] for r in task.required_roles};chosen[a]=x;chosen[b]=y
                expected=not old._hard_constraint_failures(task,chosen,by,True)
                actual=all(old.unary_score(task,c)[0]==0 for c in chosen.values()) and all(pair_compatible(task,chosen[l],chosen[r]) for l,r in edges)
                assert actual==expected,(task.task_id,a,b)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--export-audit',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.export_audit:export_audit()
    # 避免旧加载函数将输入清单写回旧结果目录。
    old.OUT=OUT
    cat,tasks,records=old.load_inputs()
    prior=old.jsonl(OLD/'matched_replay.jsonl');lookup={(r['task_id'],r['repeat'],r['method']):r for r in prior}
    rows=[];checked=set()
    for n,(tid,t) in enumerate(sorted(tasks.items()),1):
        items=[c for c in cat if c.scene==t.scene];edges=relation_edges(t)
        if t.scene not in checked:verify_pair_factorization(t,items,edges);checked.add(t.scene)
        for rep in (1,2,3):
            p=records[tid,'P',rep];start=time.perf_counter()
            ids,meta=support_pool(t,items,p['activated_ids'],edges,lambda a,b:pair_compatible(t,a,b))
            selection_ms=(time.perf_counter()-start)*1000
            start=time.perf_counter();res=old.solve_subgraph(t,ids,items);solver_ms=(time.perf_counter()-start)*1000
            rows.append(dict(task_id=tid,scene=t.scene,task_type=t.task_type,repeat=rep,method='Support',candidate_ids=list(ids),
              feasible=int(res.feasible),pool_size=len(ids),role_counts=[sum(c.role==r and c.template_id in ids for c in items) for r in t.required_roles],
              pair_checks=meta['pair_checks'],selection_ms=selection_ms,solver_ms=solver_ms,search_nodes=res.search_nodes,
              flagged=t.injected_conflict=='global_no_solution',selected=res.selected_candidate_ids))
            assert rows[-1]['role_counts']==lookup[tid,rep,'P']['role_counts']
        if n%100==0:print('offline_support_tasks',n,flush=True)
    dump(OUT/'support_replay.json',rows)
    summary=[];contrasts=[];support={(r['task_id'],r['repeat']):r for r in rows}
    for s,k in STRATA:
        tids=sorted(tid for tid,t in tasks.items() if (t.scene,t.task_type)==(s,k))
        values={m:[old.majority([support[tid,j]['feasible'] if m=='Support' else lookup[tid,j,m]['feasible'] for j in (1,2,3)]) for tid in tids]
                for m in ('P','Random','Rule','Lexical','Support','Full')}
        for m,vs in values.items():
            summary.append(dict(scene=s,task_type=k,method=m,n=len(vs),success=sum(vs),rate=sum(vs)/len(vs)))
        for m in ('Random','Rule','Lexical','Support'):
            contrasts.append(dict(scene=s,task_type=k,method_a='P',method_b=m,**old.paired(values['P'],values[m])))
    # 新版同时纳入 Support，统一 16 项 matched family；旧的 12 项结果保留于旧目录。
    old.holm(contrasts);dump(OUT/'matched_statistics_16.json',contrasts);dump(OUT/'matched_summary.json',summary)
    workload=[]
    for s,k in STRATA:
        rs=[r for r in rows if (r['scene'],r['task_type'])==(s,k)]
        workload.append(dict(scene=s,task_type=k,**{f'mean_{f}':float(np.mean([r[f] for r in rs])) for f in ('pair_checks','selection_ms','solver_ms','search_nodes','pool_size')}))
    dump(OUT/'support_workload.json',workload)
    a=json.loads((AUDIT/'p_attempts_sanitized.json').read_text(encoding='utf-8'))
    assert all(r['task_id'] in tasks and r['repeat'] in (1,2,3) for r in a)
    assert len({(r['task_id'],r['repeat'],r['stage']) for r in a})==1800
    audit=[]
    for s,k in STRATA:
        ar=[r for r in a if (tasks[r['task_id']].scene,tasks[r['task_id']].task_type)==(s,k)]
        finals=[r for (tid,m,rep),r in records.items() if m=='P' and (tasks[tid].scene,tasks[tid].task_type)==(s,k)]
        audit.append(dict(scene=s,task_type=k,**summarize_audit(ar),final_executions=len(finals),
          final_failed_executions=sum(r['status']!='success' for r in finals),
          final_record_format_failures=sum(r.get('format_failures',0) for r in finals),
          final_elapsed_seconds_quantiles=[float(v) for v in np.quantile([r['elapsed_ms']/1000 for r in finals],[.5,.9,.95])]))
    dump(OUT/'audit_summary.json',dict(strata=audit,overall=summarize_audit(a),
      scope='All retained P request attempts for 1200 task-repeat keys, including superseded task attempts; execution latency uses final selected records only. No failed run absent from this archive can be reconstructed.'))
    inputs=[OLD/'matched_replay.jsonl',AUDIT/'p_attempts_sanitized.json',AUDIT/'provenance.json']
    manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    manifest.update(analysis_date='2026-10-14',live_model_calls=0,policy='Retrospective Support extension; 16 matched Holm tests; no noninferiority or population generalization claim; no ground-truth witnesses used in ranking.')
    manifest['files'].update({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    dump(OUT/'input_manifest.json',manifest)
    print(json.dumps(dict(matched=summary,audit=audit,workload=workload),indent=2))

if __name__=='__main__':main()
