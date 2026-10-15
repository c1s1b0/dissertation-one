"""非LLM结构压力测试；显式兼容矩阵、稀有合法候选和精确前向检查。"""
from __future__ import annotations
import itertools, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reconstruct_paper2_20261013 import OUT, SEED, save, csvsave, seed_for

def edges_for(topology,roles):
    if topology=='chain': return [(i,i+1) for i in range(roles-1)]
    if topology=='star': return [(0,i) for i in range(1,roles)]
    if topology=='cycle': return [(i,i+1) for i in range(roles-1)]+[(0,roles-1)]
    if topology=='dense': return list(itertools.combinations(range(roles),2))
    raise ValueError(topology)

def generate(n,topology,density,rare,seed,roles=6):
    rng=np.random.default_rng(seed);edges=edges_for(topology,roles)
    planted=[int(rng.integers(n)) for _ in range(roles)]
    domains=[]
    for r in range(roles):
        k=max(1,int(round(n*(rare if r in (0,roles-1) else .7))))
        rest=[i for i in range(n) if i!=planted[r]]
        domains.append(sorted([planted[r],*map(int,rng.choice(rest,k-1,replace=False))]))
    mats={e:rng.random((n,n))<density for e in edges}
    for (a,b),mat in mats.items():mat[planted[a],planted[b]]=True
    return domains,mats,planted

def solve(domains,mats,node_limit=100000):
    nodes=0
    def visit(ds,assigned):
        nonlocal nodes
        if any(len(v)==0 for v in ds.values()):return None
        if not ds:return assigned
        r=min(ds,key=lambda x:(len(ds[x]),x))
        for v in ds[r]:
            nodes+=1
            if nodes>node_limit:raise TimeoutError
            reduced={}
            for s,options in ds.items():
                if s==r:continue
                if (r,s) in mats:options=[w for w in options if mats[r,s][v,w]]
                elif (s,r) in mats:options=[w for w in options if mats[s,r][w,v]]
                reduced[s]=options
            ans=visit(reduced,{**assigned,r:v})
            if ans is not None:return ans
        return None
    try:
        ans=visit({i:list(d) for i,d in enumerate(domains)},{});status='feasible' if ans is not None else 'infeasible'
    except TimeoutError:ans=None;status='unknown'
    return status,ans,nodes

def pool(n,domains,mats,k,method,seed):
    rng=np.random.default_rng(seed);out=[]
    for r in range(len(domains)):
        order=list(range(n))
        if method=='Random':rng.shuffle(order)
        elif method=='Unary':order.sort(key=lambda v:(v not in domains[r],v))
        elif method=='Support':
            def score(v):
                support=0.
                for (a,b),mat in mats.items():
                    if a==r:support+=float(mat[v,domains[b]].mean())
                    elif b==r:support+=float(mat[domains[a],v].mean())
                return (v not in domains[r],-support,v)
            order.sort(key=score)
        else:raise ValueError(method)
        out.append(sorted(set(order[:k])&set(domains[r])))
    return out

def main():
    rows=[];manifest=[];counter=0
    for n,topology,density,rare,rep in itertools.product((20,40,80),('chain','star','cycle','dense'),(.15,.60),(.05,.25),range(10)):
        seed=seed_for(SEED,n,topology,density,rare,rep);ds,ms,planted=generate(n,topology,density,rare,seed)
        status,assignment,nodes=solve(ds,ms);assert status=='feasible'
        counter+=1;case=f'S{counter:04d}'
        manifest.append(dict(case=case,n_per_role=n,roles=6,topology=topology,compatibility_probability=density,rare_fraction=rare,rep=rep,seed=seed))
        for k,method in itertools.product((3,6,12),('Random','Unary','Support')):
            active=pool(n,ds,ms,k,method,seed_for(seed,k,method));start=time.perf_counter();st,sol,nn=solve(active,ms);elapsed=(time.perf_counter()-start)*1000
            if sol is not None:assert all(ms[a,b][sol[a],sol[b]] for a,b in ms)
            rows.append(dict(case=case,n_per_role=n,roles=6,topology=topology,density=density,rare_fraction=rare,k=k,method=method,
                   status=st,witness_retained=int(st=='feasible'),pruning_fraction=1-k/n,search_nodes=nn,elapsed_ms=elapsed,
                   planted_component_recall=sum(planted[r] in active[r] for r in range(6))/6,
                   initial_candidate_count=6*k,unary_survivors=sum(map(len,active)),full_search_nodes=nodes))
        if counter%80==0:print('structural cases',counter,flush=True)
    assert counter==480 and len(rows)==4320
    csvsave('structural_tasks.csv',rows);save('structural_manifest.json',{'post_review':True,'model_calls':0,'cases':manifest,'all_planted_feasible':True,'solver_node_limit':100000,
         'scope':'abstract CSP stress test; no LLM performance, physical validity, optimality or minimum-edit claim'})
    summary=[];groups=defaultdict(list)
    for r in rows:groups[r['n_per_role'],r['topology'],r['k'],r['method']].append(r)
    for (n,top,k,met),rs in sorted(groups.items()):
        summary.append(dict(n_per_role=n,topology=top,k=k,method=met,n_tasks=len(rs),feasible_rate=sum(r['witness_retained'] for r in rs)/len(rs),
          unknown_count=sum(r['status']=='unknown' for r in rs),mean_nodes=sum(r['search_nodes'] for r in rs)/len(rs)))
    csvsave('structural_summary.csv',summary);print('DONE',counter,len(rows),'unknown',sum(r['status']=='unknown' for r in rows))

if __name__=='__main__':main()
