"""拒稿后离线重建：严格任务多数表决、角色配额匹配对照和可追溯输出。"""
from __future__ import annotations
import csv, hashlib, json, math, random, re, sys, time
from collections import defaultdict, Counter
from dataclasses import replace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paper2_dynamic_agents.experiment.schema import ComponentTemplate, TypedPort, EvidenceRecord
from paper2_dynamic_agents.experiment.mechanism import _coerce_task
from paper2_dynamic_agents.experiment.solver import solve_subgraph, _hard_constraint_failures

OUT = ROOT / 'research_artifacts/reconstruction-20261013'
BENCH = ROOT / 'research_artifacts/benchmarks/formal-20260816'
SEED = 20261013

def jsonl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def csvsave(name, rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:16],16)

def majority(values):
    if len(values)!=3: raise ValueError('exactly three repeats required')
    return int(sum(values)>=2)

def paired(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    assert len(a)==len(b) and len(a)>0
    assert set(a)<= {0.,1.} and set(b)<={0.,1.}
    d=a-b; rng=np.random.default_rng(SEED)
    means=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
    ab=int(((a==1)&(b==0)).sum()); ba=int(((a==0)&(b==1)).sum()); n=ab+ba
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(ab,ba)+1))/2**n) if n else 1.
    return dict(n=len(a),a_success=int(a.sum()),b_success=int(b.sum()),difference=float(d.mean()),
                ci95=[float(x) for x in np.quantile(means,[.025,.975])],a_only=ab,b_only=ba,p_exact=p)

def holm(rows):
    prev=0
    for i,r in enumerate(sorted(rows,key=lambda r:r['p_exact'])):
        prev=max(prev,min(1.,r['p_exact']*(len(rows)-i)));r['p_holm']=prev

def load_inputs():
    paths=[BENCH/'catalog.jsonl',BENCH/'tasks/A_test.jsonl',BENCH/'tasks/B_test.jsonl']
    cat=[]
    for d in jsonl(paths[0]):
        d['ports']=tuple(TypedPort(**x) for x in d['ports']);d['evidence']=EvidenceRecord(**d['evidence']);d['constraints']=tuple(d['constraints']);cat.append(ComponentTemplate(**d))
    tasks={d['task_id']:_coerce_task(d) for p in paths[1:] for d in jsonl(p)}
    specs=[('formal_runs_v2/test/results.jsonl',{'B0'}),
           ('formal_llm_runs/main-v5-postaudit-test-20260817/results.jsonl',{'P','P-LLM'}),
           ('formal_llm_runs/main-v5-postaudit-ablation-test-20260817/results.jsonl',{'P-Typed','P-LLMRepair'})]
    records={}
    for rel,methods in specs:
        p=ROOT/'research_artifacts'/rel;paths.append(p)
        for r in jsonl(p):
            if r['method'] in methods and r['task_id'] in tasks and r['split']=='test':
                records[r['task_id'],r['method'],r['repeat']]=r
    assert len(tasks)==400 and len(records)==6000
    assert all(r['status']=='success' for r in records.values())
    for task_id in tasks:
        for method in ('B0','P','P-LLM','P-Typed','P-LLMRepair'):
            assert {k[2] for k in records if k[:2]==(task_id,method)}=={1,2,3}
    save('input_manifest.json',{'analysis_date':'2026-10-13','seed':SEED,'bootstrap_draws':10000,
         'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
         'policy':'post-review retrospective; failures count as zero; exact 3-repeat majority; 16 primary Holm tests and 12 matched-control Holm tests separately; random replication is descriptive'})
    return cat,tasks,records

def unary_score(task,item):
    fail=0; deficit=0.
    for key,bound in task.numeric_requirements.items():
        role,field,op=key.split('.')
        if role!=item.role:continue
        v=item.parameters[field]; gap=max(0.,bound-v) if op=='min' else max(0.,v-bound)
        fail+=int(gap>0);deficit+=gap/max(1,abs(bound))
    return fail,deficit

def lexical_scores(task,items):
    # 透明词项检索：不伪装成向量检索或RAG。
    texts=[json.dumps({'role':c.role,'parameters':c.parameters},sort_keys=True) for c in items]
    query=task.requirement+' '+json.dumps(task.numeric_requirements,sort_keys=True)
    docs=[Counter(re.findall(r'[A-Za-z]+|\d+(?:\.\d+)?',s.lower())) for s in texts+[query]]
    df=Counter(w for d in docs[:-1] for w in d)
    idf={w:math.log((1+len(items))/(1+df[w]))+1 for d in docs for w in d}
    q={w:v*idf[w] for w,v in docs[-1].items()}; qn=math.sqrt(sum(v*v for v in q.values())) or 1
    ans={}
    for c,d in zip(items,docs):
        vec={w:v*idf[w] for w,v in d.items()};norm=math.sqrt(sum(v*v for v in vec.values())) or 1
        ans[c.template_id]=sum(v*q.get(w,0) for w,v in vec.items())/(norm*qn)
    return ans

def matched_pool(task,items,p_ids,kind,rep,seed):
    by={c.template_id:c for c in items};initial={x for xs in task.candidate_ids.values() for x in xs}
    assert initial<=set(p_ids)
    lex=lexical_scores(task,items) if kind=='Lexical' else {}
    result=set(initial)
    for role in task.required_roles:
        quota=sum(by[i].role==role for i in p_ids)-len(task.candidate_ids[role])
        choices=[c for c in items if c.role==role and c.template_id not in initial]
        if kind=='Random':
            rng=random.Random(seed_for(seed,task.task_id,rep,role));rng.shuffle(choices)
        elif kind=='Rule':choices.sort(key=lambda c:(*unary_score(task,c),c.template_id))
        elif kind=='Lexical':choices.sort(key=lambda c:(-lex[c.template_id],c.template_id))
        else:raise ValueError(kind)
        result.update(c.template_id for c in choices[:quota])
    assert len(result)==len(p_ids)
    for role in task.required_roles:
        assert sum(by[i].role==role for i in result)==sum(by[i].role==role for i in p_ids)
    return tuple(c.template_id for c in items if c.template_id in result)

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    cat,tasks,records=load_inputs();by={c.template_id:c for c in cat}
    grouped=defaultdict(list);replays=[];mismatches=[];example=None
    # Canonical catalog ordering removes list-order advantages from workload comparisons.
    for idx,(tid,t) in enumerate(sorted(tasks.items())):
        items=[c for c in cat if c.scene==t.scene];full_ids=[c.template_id for c in items]
        full=solve_subgraph(t,full_ids,items); full_ok=full.feasible
        for method in ('B0','P','P-LLM','P-Typed','P-LLMRepair'):
            rs=[records[tid,method,rep] for rep in (1,2,3)]
            vals=[int(r['status']=='success' and r['feasible']) for r in rs]
            grouped[t.scene,t.task_type,method].append(dict(task_id=tid,majority=majority(vals),mean=sum(vals)/3,
                full_feasible=full_ok,flagged=t.injected_conflict=='global_no_solution',
                elapsed_ms=sum(r['elapsed_ms'] for r in rs)/3,llm_calls=sum(r['llm_calls'] for r in rs)/3))
        for rep in (1,2,3):
            p=records[tid,'P',rep];canonical=tuple(c.template_id for c in items if c.template_id in p['activated_ids'])
            strategies={'P':canonical,'Initial':tuple(x for xs in t.candidate_ids.values() for x in xs),'Full':tuple(full_ids)}
            for name in ('Random','Rule','Lexical'):strategies[name]=matched_pool(t,items,canonical,name,rep,SEED)
            for name,ids in strategies.items():
                ids=tuple(c.template_id for c in items if c.template_id in ids)
                start=time.perf_counter(); res=solve_subgraph(t,ids,items);ms=(time.perf_counter()-start)*1000
                if name=='P' and res.feasible!=bool(p['feasible']):mismatches.append([tid,rep])
                counts=[sum(by[i].role==r for i in ids) for r in t.required_roles]
                row=dict(task_id=tid,scene=t.scene,task_type=t.task_type,repeat=rep,method=name,feasible=int(res.feasible),
                     full_feasible=int(full_ok),flagged=t.injected_conflict=='global_no_solution',pool_size=len(ids),
                     role_counts=counts,pruning_fraction=1-len(ids)/len(items),search_nodes=res.search_nodes,
                     solver_ms=ms,candidate_ids=list(ids),selected=res.selected_candidate_ids)
                replays.append(row)
                if example is None and name=='P' and t.scene=='B' and full_ok and res.feasible and not solve_subgraph(t,t.candidate_ids,items).feasible:
                    example=dict(task=t.to_dict(),pool=list(ids),selected=res.selected_candidate_ids,
                                 components={i:by[i].to_dict() for i in res.selected_candidate_ids.values()})
        if (idx+1)%50==0:print('matched replay',idx+1,flush=True)
    assert not mismatches, mismatches
    save('worked_example.json',example)
    with (OUT/'matched_replay.jsonl').open('w',encoding='utf-8') as f:
        for r in replays:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    primary=[];summary=[]
    for (scene,kind,method),rows in sorted(grouped.items()):
        summary.append(dict(scene=scene,task_type=kind,method=method,n=len(rows),success=sum(r['majority'] for r in rows),
                            majority_rate=sum(r['majority'] for r in rows)/len(rows),repeat_mean=sum(r['mean'] for r in rows)/len(rows),
                            historical_elapsed_ms=sum(r['elapsed_ms'] for r in rows)/len(rows),mean_llm_calls=sum(r['llm_calls'] for r in rows)/len(rows)))
    for scene in ('A','B'):
        for kind in ('configuration','repair'):
            a=grouped[scene,kind,'P']
            for method in ('B0','P-LLM','P-Typed','P-LLMRepair'):
                b=grouped[scene,kind,method];assert [r['task_id'] for r in a]==[r['task_id'] for r in b]
                primary.append(dict(scene=scene,task_type=kind,method_a='P',method_b=method,**paired([r['majority'] for r in a],[r['majority'] for r in b])))
    holm(primary);save('primary_statistics.json',primary);csvsave('primary_summary.csv',summary)
    replay_group=defaultdict(list)
    for r in replays:replay_group[r['scene'],r['task_type'],r['method'],r['task_id']].append(r)
    matched_tasks=defaultdict(list)
    for (scene,kind,m,tid),rs in sorted(replay_group.items()):
        matched_tasks[scene,kind,m].append(dict(task_id=tid,majority=majority([r['feasible'] for r in rs]),
                     full_feasible=rs[0]['full_feasible'],flagged=rs[0]['flagged'],
                     pool_size=sum(r['pool_size'] for r in rs)/3,pruning_fraction=sum(r['pruning_fraction'] for r in rs)/3,
                     search_nodes=sum(r['search_nodes'] for r in rs)/3,solver_ms=sum(r['solver_ms'] for r in rs)/3))
    matchstats=[];matchsummary=[]
    for (scene,kind,m),rs in sorted(matched_tasks.items()):
        pos=[r for r in rs if r['full_feasible']]
        matchsummary.append(dict(scene=scene,task_type=kind,method=m,n=len(rs),majority_success=sum(r['majority'] for r in rs),
             majority_rate=sum(r['majority'] for r in rs)/len(rs),full_feasible_n=len(pos),
             witness_retention=sum(r['majority'] for r in pos)/len(pos),witness_loss=1-sum(r['majority'] for r in pos)/len(pos),
             **{f'mean_{key}':sum(r[key] for r in rs)/len(rs) for key in ('pool_size','pruning_fraction','search_nodes','solver_ms')}))
    for scene in ('A','B'):
        for kind in ('configuration','repair'):
            a=matched_tasks[scene,kind,'P']
            for method in ('Random','Rule','Lexical'):
                b=matched_tasks[scene,kind,method]
                matchstats.append(dict(scene=scene,task_type=kind,method_a='P',method_b=method,**paired([r['majority'] for r in a],[r['majority'] for r in b])))
    holm(matchstats);save('matched_statistics.json',matchstats);csvsave('matched_summary.csv',matchsummary)
    # All-one typed interfaces are audited rather than interpreted as an informative typed ablation.
    port_profiles={s:sorted({tuple((p.port_type,p.direction,p.unit) for p in c.ports) for c in cat if c.scene==s}) for s in ('A','B')}
    save('structural_audit.json',{'p_replay_mismatches':mismatches,'flagged_no_solution_tasks':sum(t.injected_conflict=='global_no_solution' for t in tasks.values()),'port_profiles':port_profiles,
         'repair_has_fixed_initial_assignment':False,'historical_primary_test':'Wilcoxon on within-task means; McNemar stored separately',
         'matched_is_posthoc':True,'random_seed':SEED,'live_api_calls':0})
    print(json.dumps({'primary':summary,'matched':matchsummary},indent=2))

if __name__=='__main__':main()
