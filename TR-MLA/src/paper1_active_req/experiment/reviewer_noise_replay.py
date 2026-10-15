"""固定历史查询顺序的含噪输入保护测试，不是自适应重规划或纠错系统。"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from paper1_active_req.domain.models import FieldKind, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
from paper1_active_req.evaluation.v3_metrics import _matches_truth
from paper1_active_req.experiment.v3_episode import V3EpisodeResult, V3EpisodeStep
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine

POLICIES = ('criticality_greedy', 'marginal_lookahead_no_relations', 'typed_relational_myopic', 'tr_mla')
GROUPS = ((0., 0.), (.1, .05), (.1, .10), (.2, .05), (.2, .10))
SEED = 20261009


def noisy_answer(original, definition, spec_id, probability, amplitude):
    """正常回答不扰动；按规格/字段共用概率抽样、方向和类别选择。"""
    answer = dict(original)
    digest = hashlib.sha256(f'{SEED}:{spec_id}:{definition.field_id}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    draw = float(rng.random())
    contaminated = draw < probability and answer['value'] is not None
    if contaminated:
        if definition.kind == FieldKind.BOOLEAN:
            answer['value'] = not bool(answer['value'])
        elif definition.kind == FieldKind.CATEGORICAL:
            options = [x for x in definition.categories if x != answer['value']]
            answer['value'] = options[int(rng.integers(len(options)))]
        else:
            value = float(answer['value']) * (1 + amplitude * int(rng.choice((-1, 1))))
            answer['value'] = value if definition.kind == FieldKind.CONTINUOUS else int(round(value))
        answer['source'] = 'reviewer_noise'
    out_of_domain = False
    if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER} and answer['value'] is not None:
        out_of_domain = not definition.lower <= answer['value'] <= definition.upper
    return {'answer': answer, 'contaminated': contaminated, 'out_of_domain': out_of_domain, 'probability_draw': draw}


def apply_candidate(state, field_id, answer, engine, guard):
    candidate = state.apply_answer(field_id, FieldValue.model_validate(answer))
    checked = engine.check_satisfiability(candidate)
    accepted = not guard or checked.satisfiable is True
    return (candidate if accepted else state.public_copy()), {
        'accepted': accepted, 'candidate_sat': checked.satisfiable,
        'candidate_unknown': checked.satisfiable is None, 'budget_consumed': 1}


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def _csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_noise_replay(root: Path, output_dir: Path, episode_limit=None):
    inputs = {'formal_episodes': root/'results/v3/formal/episodes.jsonl',
              'tasks': root/'data/generated/v3/test_tasks.jsonl',
              'train': root/'data/generated/v3/train_complete_specs.jsonl',
              'schema': root/'configs/fields.yaml', 'relations': root/'configs/v3_relations.yaml'}
    schema = load_field_schema(inputs['schema'])
    config = load_v3_relation_config(inputs['relations'])
    model = TypedRelationalModel.fit([CompleteSpec.model_validate(x) for x in _read(inputs['train'])], schema, config)
    tasks = {x['task_id']: V3RequirementTask.model_validate(x) for x in _read(inputs['tasks'])}
    original = [x for x in _read(inputs['formal_episodes']) if x['policy_name'] in POLICIES]
    if episode_limit is not None:
        original = original[:episode_limit]
    engine = ConstraintEngine(schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 独立目录保留审计记录；已存在的文件禁止覆盖。
    record_path = output_dir/'episodes.jsonl'
    if record_path.exists():
        raise ValueError('noise replay output already exists; refusing to overwrite')
    start = time.perf_counter()
    rows = []
    with record_path.open('x', encoding='utf-8') as handle:
        for original_index, record in enumerate(original):
            task = tasks[record['task_id']]
            episode = V3EpisodeResult.model_validate(record['episode'])
            clean_reference = reviewer_metrics(episode, task, schema, model)
            for probability, amplitude in GROUPS:
                shared = [noisy_answer(step.answer.model_dump(mode='json'), schema[step.field_id],
                                       task.spec_id, probability, amplitude) for step in episode.steps[:8]]
                for guard in (False, True):
                    state = episode.initial_state.public_copy()
                    events, effective_steps = [], []
                    for step, noise in zip(episode.steps[:8], shared, strict=True):
                        before = state.public_copy()
                        state, event = apply_candidate(state, step.field_id, noise['answer'], engine, guard)
                        effective_answer = (FieldValue.model_validate(noise['answer']) if event['accepted']
                                            else before.values[step.field_id].model_copy(deep=True))
                        effective_steps.append(V3EpisodeStep(step_index=step.step_index, field_id=step.field_id,
                                                            answer=effective_answer, satisfiable_after_answer=event['candidate_sat']))
                        semantic_correct = _matches_truth(noise['answer']['value'], task.ground_truth[step.field_id], schema[step.field_id])
                        events.append({'step_index': step.step_index, 'field_id': step.field_id,
                                       'original_answer': step.answer.model_dump(mode='json'),
                                       'truth': task.ground_truth[step.field_id].model_dump(mode='json'),
                                       'new_answer': noise['answer'], 'effective_answer': effective_answer.model_dump(mode='json'),
                                       **{key: value for key, value in noise.items() if key != 'answer'}, **event,
                                       'semantic_correct': semantic_correct,
                                       'wrong_but_sat_accepted': not semantic_correct and event['accepted'] and event['candidate_sat'] is True})
                    replay = V3EpisodeResult(task_id=episode.task_id, policy_name=episode.policy_name,
                                             initial_state=episode.initial_state, final_state=state,
                                             steps=effective_steps, solver_calls=len(effective_steps)+1)
                    metrics = reviewer_metrics(replay, task, schema, model)
                    final_sat = engine.check_satisfiability(state).satisfiable
                    clean_match = all(abs(metrics[key]-clean_reference[key]) <= 1e-12 for key in metrics) if probability == 0 else None
                    result = {'task_id': task.task_id, 'spec_id': task.spec_id, 'policy_name': episode.policy_name,
                              'missing_rate': task.missing_rate, 'error_probability': probability,
                              'continuous_error_amplitude': amplitude, 'guard': guard,
                              'contaminated_answer_count': sum(x['contaminated'] for x in events),
                              'rejected_answer_count': sum(not x['accepted'] for x in events),
                              'wrong_but_sat_accepted_count': sum(x['wrong_but_sat_accepted'] for x in events),
                              'out_of_domain_answer_count': sum(x['out_of_domain'] for x in events),
                              'unknown_candidate_count': sum(x['candidate_unknown'] for x in events),
                              'final_sat': final_sat, 'final_unknown': final_sat is None,
                              'actual_question_count': len(events), 'clean_metric_matches_original_reviewer': clean_match,
                              'metrics': metrics, 'steps': events, 'final_state': state.model_dump(mode='json')}
                    handle.write(json.dumps(result, ensure_ascii=False)+'\n')
                    rows.append({key: value for key, value in result.items() if key not in {'metrics','steps','final_state'}} |
                                {'dcrr_at_8': metrics['dcrr_at_8'], 'rcrr_at_8': metrics['rcrr_at_8'],
                                 'rcrr_at_0': metrics['rcrr_at_0'], 'answer_cost': metrics['answer_cost']})
            if (original_index+1) % 180 == 0:
                print(json.dumps({'event':'noise_replay_progress', 'original_done':original_index+1,
                                  'replay_done':len(rows)}, ensure_ascii=False), flush=True)
    _csv(output_dir/'episode_metrics.csv', rows)
    aggregate = []
    for policy in POLICIES:
        for probability, amplitude in GROUPS:
            for guard in (False, True):
                group = [x for x in rows if (x['policy_name'], x['error_probability'], x['continuous_error_amplitude'], x['guard']) ==
                         (policy, probability, amplitude, guard)]
                if not group:
                    continue
                aggregate.append({'policy_name': policy, 'error_probability': probability,
                                  'continuous_error_amplitude': amplitude, 'guard': guard, 'episode_n': len(group),
                                  'spec_n': len({x['spec_id'] for x in group}),
                                  **{f'total_{key}': sum(x[key] for x in group) for key in
                                     ('contaminated_answer_count', 'rejected_answer_count', 'wrong_but_sat_accepted_count',
                                      'out_of_domain_answer_count', 'unknown_candidate_count')},
                                  'final_sat_rate': sum(x['final_sat'] is True for x in group)/len(group),
                                  'final_unknown_n': sum(x['final_unknown'] for x in group),
                                  **{f'mean_{key}': float(np.mean([x[key] for x in group])) for key in
                                     ('dcrr_at_8','rcrr_at_8','rcrr_at_0','answer_cost','actual_question_count')}})
    _csv(output_dir/'summary.csv', aggregate)
    summary = {'original_episode_count': len(original), 'replay_count': len(rows), 'seed': SEED,
               'groups': GROUPS, 'policies': POLICIES, 'budget': 8, 'train_n': 180,
               'runtime_seconds': time.perf_counter()-start,
               'clean_metric_mismatch_n': sum(x['clean_metric_matches_original_reviewer'] is False for x in rows),
               'input_hashes': {key: {'path':str(path.resolve()), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in inputs.items()},
               'source_hashes': {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), root/'src/paper1_active_req/evaluation/reviewer_metrics.py',
                                             root/'src/paper1_active_req/solver/engine.py', root/'src/paper1_active_req/solver/rules.py')},
               'summary': aggregate,
               'limitations': ['固定历史查询顺序的输入保护测试，不是自适应含噪重规划',
                               '拒绝后消耗预算但保留前状态、不重问、不恢复正确答案',
                               '约束可满足不等于语义正确，错误但SAT接受数单独报告',
                               '不是完整纠错系统或自然语言噪声工业验证',
                               '各缺失率副本不是独立规格；此表为描述性，无显著性推断']}
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
