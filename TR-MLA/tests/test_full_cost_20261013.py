"""全网格与记录完整性测试，避免复用或缺格静默污染统计。"""
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_full_cost_20261013.py'

def module():
    assert SCRIPT.exists(), '独立全量运行器尚未实现'
    spec = importlib.util.spec_from_file_location('full_cost', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def test_full_grid_and_exact_reuse():
    m = module()
    jobs = m.plan()
    assert len(jobs) == len({j['key'] for j in jobs}) == 4320
    old = m.load_old()
    reused = [r for r in old if r['key'] in {j['key'] for j in jobs}]
    assert len(reused) == 1404
    assert sum(r['lambda_cost'] == 0 for r in reused) == 1080

def test_missing_duplicate_and_failed_are_rejected():
    m = module()
    expected = [{'key':'a','seed':1}]
    good = {'key':'a','seed':1,'status':'success','protocol_hash':'x'}
    m.validate_coverage([good], expected, 'x')
    for bad in [[], [good,good], [{**good,'status':'failed'}], [{**good,'seed':2}]]:
        with pytest.raises(ValueError):
            m.validate_coverage(bad, expected, 'x')

def test_scope_aggregation_preserves_spec_pairs():
    path = SCRIPT.with_name('summarize_full_cost_20261013.py')
    assert path.exists(), '全量统计器尚未实现'
    spec = importlib.util.spec_from_file_location('full_summary',path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import pandas as pd
    frame = pd.DataFrame([dict(seed=1,spec_id=s,policy_name='p',lambda_cost=0.,missing_rate=rate,
        **{metric:value for metric in m.METRICS})
        for s,rate,value in [('a',.3,99),('a',.5,2),('a',.7,4),('b',.5,10),('b',.7,20)]])
    result=m.scope_specs(frame,'m50_m70_mean').set_index('spec_id')
    assert result.loc['a','answer_cost']==3
    assert result.loc['b','answer_cost']==15
    with pytest.raises(ValueError): m.scope_specs(frame.iloc[:-1],'m50_m70_mean')
