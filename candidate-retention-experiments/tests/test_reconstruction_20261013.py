"""重构版统计与新增离线对照的独立数学核验。"""
import itertools, sys
from pathlib import Path
import numpy as np
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reconstruct_paper2_20261013 import majority,paired,holm
from structural_stress_20261013 import generate,solve,pool

def test_majority_not_repeat_mean_and_reject_incomplete():
    assert majority([1,1,0])==1
    assert majority([1,0,0])==0
    with unittest.TestCase().assertRaises(ValueError):majority([1,0])

def test_exact_mcnemar_matches_known_binomial_tail():
    r=paired([1]*5,[0]*5)
    assert r['p_exact']==.0625 and r['difference']==1
    assert r['ci95']==[1,1] and r['a_only']==5 and r['b_only']==0
    assert paired([1,0],[0,1])['p_exact']==1
    with unittest.TestCase().assertRaises(AssertionError):paired([1/3],[0])

def test_holm_known_family():
    rows=[{'p_exact':p} for p in (.01,.04,.03)]
    holm(rows)
    assert [r['p_holm'] for r in rows]==[.03,.06,.06]

def check_forward_checker_against_exhaustive_oracle(topology):
    for seed in range(6):
        ds,ms,plant=generate(4,topology,.3,.5,seed,roles=4)
        for method in ('Random','Unary','Support'):
            active=pool(4,ds,ms,2,method,seed)
            truth=any(all(m[x[a],x[b]] for (a,b),m in ms.items()) for x in itertools.product(*active))
            status,assignment,nodes=solve(active,ms)
            assert (status=='feasible')==truth
            if assignment is not None:assert all(m[assignment[a],assignment[b]] for (a,b),m in ms.items())

def test_exhaustion_not_inferred_from_timeout():
    ds=[[0,1],[0,1]];ms={(0,1):np.zeros((2,2),dtype=bool)}
    assert solve(ds,ms)[0]=='infeasible'
    assert solve(ds,ms,node_limit=0)[0]=='unknown'

def test_forward_checker_all_topologies():
    for topology in ('chain','star','cycle','dense'):
        check_forward_checker_against_exhaustive_oracle(topology)

if __name__=='__main__':
    tests=[v for k,v in list(globals().items()) if k.startswith('test_') and callable(v)]
    suite=unittest.TestSuite(unittest.FunctionTestCase(f) for f in tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
