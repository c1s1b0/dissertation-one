"""审稿后双指标；独立重放，不修改历史精确阈值和冻结关键集。"""
from __future__ import annotations

from collections.abc import Mapping
from math import isclose

from paper1_active_req.domain.models import FieldDefinition, FieldKind, FieldStatus
from paper1_active_req.evaluation.v3_metrics import ReliableInferenceModel, _matches_truth
from paper1_active_req.evaluation.v3_protocol import frozen_critical_field_ids
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask


def reviewer_metrics(
    episode: V3EpisodeResult,
    task: V3RequirementTask,
    schema: Mapping[str, FieldDefinition],
    model: ReliableInferenceModel,
) -> dict[str, float]:
    """返回预算0至8的DCRR/RCRR及预算8的关键集推断诊断。

    AUC按梯形法除以8；不足8问的轨迹末状态保持。可靠推断由调用者
    提供的共同模型确定。精确匹配复用历史绝对1e-9定义；容差正确仅作
    诊断，连续字段采用model.relation_config.engineering_tolerances，
    未配置容差的字段仍用精确匹配。推断数仅覆盖冻结关键集，直接正确
    字段不重复计为推断；没有可靠推断时精确率为0，需结合分母解读。
    """
    if episode.task_id != task.task_id:
        raise ValueError('episode result and task_id must match')
    if set(schema) != set(task.ground_truth):
        raise ValueError('schema and task ground truth fields must match')
    if episode.initial_state != task.initial_state:
        raise ValueError('episode and task initial states must match')
    critical_ids = frozen_critical_field_ids(task.initial_state, schema)
    denominator = sum(schema[key].criticality for key in critical_ids)
    if denominator <= 0:
        raise ValueError('DCRR/RCRR requires positive total criticality')
    tolerances = getattr(getattr(model, 'relation_config', None), 'engineering_tolerances', {})
    state = episode.initial_state.public_copy()
    direct_curve, recovery_curve = [], []
    metrics: dict[str, float] = {}
    for budget in range(9):
        if budget and budget <= len(episode.steps):
            step = episode.steps[budget - 1]
            state = state.apply_answer(step.field_id, step.answer)
        inferred = model.reliably_resolved(state.public_copy())
        direct_weight = recovered_weight = 0.0
        total = correct = tolerant = continuous = continuous_correct = continuous_tolerant = 0
        for key in critical_ids:
            definition, truth = schema[key], task.ground_truth[key]
            if key not in state.values:
                raise ValueError(f'task/state missing schema field: {key}')
            value = state.values[key]
            direct = value.status == FieldStatus.OBSERVED and _matches_truth(value.value, truth, definition)
            exact = False
            if not direct and key in inferred:
                total += 1
                exact = _matches_truth(inferred[key], truth, definition)
                correct += int(exact)
                within_tolerance = exact
                if definition.kind == FieldKind.CONTINUOUS:
                    continuous += 1
                    if key in tolerances and isinstance(inferred[key], (int, float)) and isinstance(truth.value, (int, float)):
                        within_tolerance = isclose(float(inferred[key]), float(truth.value), rel_tol=0, abs_tol=tolerances[key])
                    continuous_correct += int(exact)
                    continuous_tolerant += int(within_tolerance)
                tolerant += int(within_tolerance)
            direct_weight += definition.criticality * int(direct)
            recovered_weight += definition.criticality * int(direct or exact)
        direct_curve.append(direct_weight / denominator)
        recovery_curve.append(recovered_weight / denominator)
        metrics[f'dcrr_at_{budget}'] = direct_curve[-1]
        metrics[f'rcrr_at_{budget}'] = recovery_curve[-1]
        if budget == 8:
            metrics.update(inference_total=float(total), inference_exact_correct=float(correct),
                           inference_tolerance_correct=float(tolerant), inference_continuous_total=float(continuous),
                           inference_continuous_exact_correct=float(continuous_correct),
                           inference_continuous_tolerance_correct=float(continuous_tolerant),
                           inference_precision=correct / total if total else 0.0,
                           inference_tolerance_precision=tolerant / total if total else 0.0)
    metrics['dcrr_auc'] = sum((a + b) / 2 for a, b in zip(direct_curve, direct_curve[1:])) / 8
    metrics['rcrr_auc'] = sum((a + b) / 2 for a, b in zip(recovery_curve, recovery_curve[1:])) / 8
    metrics['recovery_increment'] = recovery_curve[8] - recovery_curve[0]
    metrics['answer_cost'] = float(sum(schema[step.field_id].answer_cost for step in episode.steps[:8]))
    metrics['actual_question_count'] = float(len(episode.steps[:8]))
    return metrics
