"""候选/解集合召回、随机种子敏感性与真实单任务数值核查。"""
import itertools,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_paper2_20261013 import *

def all_witnesses(t,items):
    roles=t.required_roles;by={r:[c for c in items if c.role==r] for r in roles}
    grid=np.array(list(itertools.product(range(20),repeat=4)),dtype=np.int16)
    values={r:{f:np.array([c.parameters[f] for c in by[r]])[grid[:,j]] for f in by[r][0].parameters} for j,r in enumerate(roles)}
    valid=np.ones(len(grid),dtype=bool)
    for key,b in t.numeric_requirements.items():
        r,f,op=key.split('.');valid &= values[r][f]>=b if op=='min' else values[r][f]<=b
    if t.scene=='A':
        valid &= values['drive']['capacity_kg']>=values['conveyor']['load_kg']
        valid &= values['positioner']['payload_kg']>=values['conveyor']['load_kg']
        valid &= values['transmission']['speed_rpm']<=2*values['drive']['speed_rpm']
    else:
        valid &= values['gearbox']['input_speed_rpm']<=values['motor']['speed_rpm']
        valid &= values['coupling']['torque_nm']>=values['motor']['torque_nm']
        valid &= values['coupling']['bore_mm']>=values['motor']['shaft_diameter_mm']
        valid &= values['shaft_support']['bore_mm']>=values['motor']['shaft_diameter_mm']
    if t.injected_conflict=='global_no_solution':valid[:]=False
    return grid[valid],by

def main():
    cat,tasks,records=load_inputs();replay=jsonl(OUT/'matched_replay.jsonl');group=defaultdict(list)
    for r in replay:group[r['task_id']].append(r)
    metrics=[]
    for i,(tid,t) in enumerate(sorted(tasks.items())):
        items=[c for c in cat if c.scene==t.scene];w,by=all_witnesses(t,items)
        if not len(w):continue
        full_solution_counts={r.get('full_catalog_solution_count') for (task,method,rep),r in records.items() if task==tid and method=='B0'}
        assert full_solution_counts=={len(w)},(tid,full_solution_counts,len(w))
        relevant=[set(w[:,j].tolist()) for j in range(4)]
        for r in group[tid]:
            active=set(r['candidate_ids']);sel=[set(j for j,c in enumerate(by[role]) if c.template_id in active) for role in t.required_roles]
            retained=np.ones(len(w),dtype=bool)
            for j in range(4):retained &= np.isin(w[:,j],list(sel[j]))
            assert bool(retained.any())==bool(r['feasible'])
            metrics.append(dict(task_id=tid,scene=t.scene,task_type=t.task_type,repeat=r['repeat'],method=r['method'],
               feasible_candidate_recall=sum(len(relevant[j]&sel[j]) for j in range(4))/sum(map(len,relevant)),
               solution_fraction=float(retained.mean()),full_solution_count=len(w),retained_solution_count=int(retained.sum())))
        if (i+1)%100==0:print('witness sets',i+1,flush=True)
    csvsave('candidate_recall.csv',metrics)
    summaries=[];groups=defaultdict(list)
    for r in metrics:groups[r['scene'],r['task_type'],r['method']].append(r)
    for (s,t,m),rs in sorted(groups.items()):summaries.append(dict(scene=s,task_type=t,method=m,n_tasks=len(rs)//3,
         mean_feasible_candidate_recall=sum(r['feasible_candidate_recall'] for r in rs)/len(rs),mean_solution_fraction=sum(r['solution_fraction'] for r in rs)/len(rs)))
    csvsave('candidate_recall_summary.csv',summaries)
    random_summary=[];random_rows=[]
    for extra in range(1,10):
        gg=defaultdict(list)
        for tid,t in sorted(tasks.items()):
            items=[c for c in cat if c.scene==t.scene];vals=[]
            for rep in (1,2,3):
                pool_ids=matched_pool(t,items,records[tid,'P',rep]['activated_ids'],'Random',rep,SEED+extra)
                val=int(solve_subgraph(t,pool_ids,items).feasible);vals.append(val)
                random_rows.append(dict(seed=SEED+extra,task_id=tid,repeat=rep,feasible=val,candidate_ids=pool_ids))
            gg[t.scene,t.task_type].append(majority(vals))
        for (s,t),vs in sorted(gg.items()):random_summary.append(dict(seed=SEED+extra,scene=s,task_type=t,n=100,success=sum(vs),rate=sum(vs)/100))
        print('random seed',extra,flush=True)
    csvsave('random_seed_summary.csv',random_summary);save('random_seed_records.json',random_rows)
    ex=json.loads((OUT/'worked_example.json').read_text(encoding='utf-8'))
    print(json.dumps({i:d['parameters'] for i,d in ex['components'].items()},indent=2))

if __name__=='__main__':main()
