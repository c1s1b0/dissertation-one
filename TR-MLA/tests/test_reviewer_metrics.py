"""双指标边界测试：推断计分、错误答案及冻结分母的改变均应被发现。"""
from types import SimpleNamespace

import pytest

from test_v3_metrics import _task_and_schema, _episode, _value, StaticInferenceModel
from paper1_active_req.domain.models import FieldKind, FieldStatus


def test_direct_recall_excludes_inference_and_replays_short_episode():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    metrics = reviewer_metrics(_episode(task), task, schema, StaticInferenceModel({'pivot': 'left'}))
    assert metrics['dcrr_at_0'] == 0
    assert metrics['rcrr_at_0'] == 0.5
    assert metrics['dcrr_at_8'] == 0.5
    assert metrics['rcrr_at_8'] == 1
    assert metrics['dcrr_auc'] == pytest.approx(0.46875)
    assert metrics['rcrr_auc'] == pytest.approx(0.96875)
    assert metrics['recovery_increment'] == 0.5
    assert metrics['answer_cost'] == 2
    assert metrics['actual_question_count'] == 1
    assert metrics['inference_total'] == 1
    assert metrics['inference_exact_correct'] == 1


def test_wrong_answer_and_wrong_inference_get_no_credit():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    episode = _episode(task)
    episode.steps[0].answer = _value('no', FieldStatus.OBSERVED)
    metrics = reviewer_metrics(episode, task, schema, StaticInferenceModel({'pivot': 'right'}))
    assert metrics['dcrr_at_8'] == metrics['rcrr_at_8'] == 0
    assert metrics['inference_exact_correct'] == 0


def test_continuous_engineering_tolerance_does_not_change_exact_recovery():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    schema['pivot'] = schema['pivot'].model_copy(update={'kind': FieldKind.CONTINUOUS, 'lower': 0, 'upper': 100})
    task.ground_truth['pivot'] = _value(10, FieldStatus.OBSERVED)
    model = StaticInferenceModel({'pivot': 10.05})
    model.relation_config = SimpleNamespace(engineering_tolerances={'pivot': 0.1})
    metrics = reviewer_metrics(_episode(task), task, schema, model)
    assert metrics['rcrr_at_8'] == 0.5
    assert metrics['inference_continuous_total'] == 1
    assert metrics['inference_exact_correct'] == 0
    assert metrics['inference_tolerance_correct'] == 1
    model.inferred['pivot'] = 10 + 1e-10
    assert reviewer_metrics(_episode(task), task, schema, model)['inference_exact_correct'] == 1


def test_correct_direct_observation_is_not_counted_as_inference():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    metrics = reviewer_metrics(_episode(task), task, schema, StaticInferenceModel({'critical': 'yes'}))
    assert metrics['inference_total'] == 0
    assert metrics['inference_precision'] == 0


def test_empty_frozen_critical_set_raises():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    schema = {key: value.model_copy(update={'criticality': 0}) for key, value in schema.items()}
    with pytest.raises(ValueError, match='positive'):
        reviewer_metrics(_episode(task), task, schema, StaticInferenceModel({}))


def test_task_binding_and_truth_schema_are_validated():
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    task, schema = _task_and_schema()
    episode = _episode(task)
    episode.task_id = 'wrong'
    with pytest.raises(ValueError, match='task_id'):
        reviewer_metrics(episode, task, schema, StaticInferenceModel({}))
    episode.task_id = task.task_id
    with pytest.raises(ValueError, match='ground truth'):
        reviewer_metrics(episode, task, {'critical': schema['critical']}, StaticInferenceModel({}))


def test_historical_pairing_averages_m50_m70_and_keeps_negative_results():
    from scripts.audit_reviewer_metrics import paired_summary
    import pandas as pd
    rows = []
    for spec in range(3):
        for method, value in [('tr_mla', 0.5), ('criticality_greedy', 0.75), ('relational_myopic', 0.8)]:
            for rate in [0.3, 0.5, 0.7]:
                rows.append({'spec_id': str(spec), 'policy_name': method, 'missing_rate': rate,
                             'crr_at_8': value if rate != 0.3 else 1,
                             'dcrr_at_8': value if rate != 0.3 else 1})
    summary = paired_summary(pd.DataFrame(rows), 20260806, ['dcrr_at_8'])
    criticality = summary[summary.baseline == 'criticality_greedy'].iloc[0]
    assert criticality.n_specs == 3
    assert criticality.mean_difference == -0.25
    assert criticality.ci_low == criticality.ci_high == -0.25
    assert criticality.bootstrap_replicates == 10000
    assert (summary.holm_p_value >= summary.p_value).all()


def test_historical_pairing_rejects_missing_missingness_replica():
    from scripts.audit_reviewer_metrics import paired_summary
    import pandas as pd
    with pytest.raises(ValueError, match='both m50 and m70'):
        paired_summary(pd.DataFrame([{'spec_id': 's', 'policy_name': 'tr_mla',
                                     'missing_rate': 0.5, 'crr_at_8': 0.5}]), 1, ['crr_at_8'])
