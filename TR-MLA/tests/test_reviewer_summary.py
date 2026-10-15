"""补充实验统计契约：完整性门槛、规格级配对、固定成本网格。"""
import pandas as pd
import pytest


def _primary_frame():
    from scripts.summarize_reviewer_supplement import SEEDS, POLICIES, PAIRED_METRICS
    rows = []
    for seed in SEEDS:
        for spec in range(60):
            for policy in POLICIES:
                for rate in [.3, .5, .7]:
                    value = 1 if rate == .3 else (.4 if policy == 'tr_mla' else .6)
                    rows.append(dict(seed=seed, spec_id=f's{spec}', policy_name=policy,
                                     missing_rate=rate, lambda_cost=0., group='multiseed',
                                     mechanism=f'm{spec // 20}', pivot=spec % 20 < 15,
                                     cost_subset=spec % 20 < 6 and rate == .7,
                                     actual_question_count=8., **{metric: value for metric in PAIRED_METRICS}))
    return pd.DataFrame(rows)


def test_primary_statistics_uses_60_specs_and_holm_18_tests_per_seed():
    from scripts.summarize_reviewer_supplement import primary_statistics
    specs, means, comparisons, cross = primary_statistics(_primary_frame())
    assert len(specs) == 3 * 60 * 4
    assert len(comparisons) == 3 * 18
    assert (comparisons.n_specs == 60).all()
    assert comparisons.mean_difference.tolist() == pytest.approx([-.2] * 54)
    assert (comparisons.holm_p_value >= comparisons.p_value).all()
    assert (means.n_specs == 60).all()
    assert (cross.seed_count == 3).all()


def test_missing_replica_is_an_error_instead_of_silent_drop():
    from scripts.summarize_reviewer_supplement import primary_statistics
    frame = _primary_frame()
    with pytest.raises(ValueError, match='m50.*m70'):
        primary_statistics(frame.drop(frame[(frame.seed == 20261011) & (frame.spec_id == 's0') &
                                            (frame.policy_name == 'tr_mla') & (frame.missing_rate == .7)].index))


def test_complete_gate_rejects_failed_status_hash_and_missing_keys():
    from scripts.summarize_reviewer_supplement import validate_records
    expected = {'a': {'key': 'a', 'seed': 1}, 'b': {'key': 'b', 'seed': 1}}
    records = [dict(key=key, seed=1, status='success', protocol_hash='p') for key in expected]
    outcome = dict(status='complete', protocol_hash='p', records_hash='h', record_count=2, expected_count=2)
    validate_records(records, expected, 'p', outcome, 'h')
    with pytest.raises(ValueError, match='complete'):
        validate_records(records, expected, 'p', {**outcome, 'status': 'running'}, 'h')
    with pytest.raises(ValueError, match='hash'):
        validate_records(records, expected, 'p', outcome, 'other')
    with pytest.raises(ValueError, match='coverage'):
        validate_records(records[:1], expected, 'p', outcome, 'h')
    records[0]['status'] = 'failed'
    with pytest.raises(ValueError, match='failed'):
        validate_records(records, expected, 'p', outcome, 'h')


def test_complete_gate_rejects_metadata_tampering_and_duplicate_keys():
    from scripts.summarize_reviewer_supplement import validate_records
    expected = {'a': {'key': 'a', 'seed': 1}}
    row = dict(key='a', seed=2, status='success', protocol_hash='p')
    outcome = dict(status='complete', protocol_hash='p', records_hash='h', record_count=1, expected_count=1)
    with pytest.raises(ValueError, match='metadata'):
        validate_records([row], expected, 'p', outcome, 'h')
    row['seed'] = 1
    with pytest.raises(ValueError, match='duplicate'):
        validate_records([row, row], expected, 'p', outcome, 'h')


def test_cost_summary_keeps_all_lambdas_and_reuses_zero_subset():
    from scripts.summarize_reviewer_supplement import cost_statistics
    frame = _primary_frame()
    zero = frame[frame.cost_subset & frame.policy_name.isin(['tr_mla', 'typed_relational_myopic'])]
    extras = [zero.assign(lambda_cost=lam, group='cost') for lam in [.05, .1, .2]]
    details, means, paired = cost_statistics(pd.concat([frame, *extras], ignore_index=True))
    assert len(details) == 3 * 18 * 2 * 4
    assert len(means) == 3 * 2 * 4
    assert set(means.lambda_cost) == {0, .05, .1, .2}
    assert (means.n_specs == 18).all()
    assert len(paired) == 3 * 4 * 6
    with pytest.raises(ValueError, match='grid'):
        cost_statistics(pd.concat([frame, *extras[:-1]], ignore_index=True))
