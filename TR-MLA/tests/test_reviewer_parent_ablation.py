import importlib
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def module():
    try:
        return importlib.import_module('paper1_active_req.experiment.reviewer_parent_ablation')
    except ModuleNotFoundError:
        pytest.fail('缺少独立多父变量诊断模块')

def test_paired_error_stats_and_holm_are_spec_level():
    m = module()
    stats = m.paired_error_stats([3.,4.,5.], [1.,2.,3.], seed=20261009)
    assert stats['n_specs']==3
    assert stats['mean_absolute_error_improvement']==2
    assert stats['bootstrap_ci95']==[2.,2.]
    assert 0 <= stats['wilcoxon_p'] <= 1
    assert m.holm_adjust([.01,.02,.20])==pytest.approx([.03,.04,.20])

def test_protocol_is_written_before_any_model_fit(tmp_path, monkeypatch):
    m = module()
    def fail_fit(*args, **kwargs):
        protocol = json.loads((tmp_path/'protocol.json').read_text(encoding='utf-8'))
        assert protocol['bootstrap_repetitions']==10000
        assert protocol['seeds']==[20261011,20261013,20261017]
        assert protocol['confidence_threshold']==.95
        raise RuntimeError('protocol-before-fit checked')
    monkeypatch.setattr(m.TypedRelationalModel, 'fit', fail_fit)
    with pytest.raises(RuntimeError, match='protocol-before-fit checked'):
        m.run_parent_ablation(ROOT, tmp_path)

def test_full_ablation_exports_720_rows_six_comparisons_no_target_leak(tmp_path):
    m = module()
    result = m.run_parent_ablation(ROOT, tmp_path)
    assert result['prediction_count']==720
    assert len(result['comparisons'])==6
    assert all(x['n_specs']==60 for x in result['comparisons'])
    assert all(0<=x['holm_p']<=1 for x in result['comparisons'])
    import csv
    with (tmp_path/'predictions.csv').open(encoding='utf-8-sig') as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==720
    for row in rows:
        public=json.loads(row['public_state'])
        assert public['values'][row['target']]['value'] is None
        assert sum(v['value'] is not None for v in public['values'].values())==2
        assert row['parent_count'] in {'1','2'}
        assert row['parameter_count'] in {'2','3'}
