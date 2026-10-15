"""对全量已完成轨迹独立只读重放、真值、可行性和双指标复验。"""
from pathlib import Path
import json
import math
from paper1_active_req.experiment.reviewer_runner import SEEDS, sha256, write_once
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.domain.models import FieldStatus
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine
from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
from paper1_active_req.evaluation.v3_metrics import _matches_truth
from summarize_reviewer_supplement import load_complete_records


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / 'research_artifacts/reviewer_supplement_20261009'
    frame,binding = load_complete_records(root,project)
    path = root / 'records/episodes.jsonl'
    before = sha256(path)
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    schema = load_field_schema(project / 'configs/fields.yaml')
    config = load_v3_relation_config(project / 'configs/v3_relations.yaml')
    checked = steps = numeric_values = 0
    for seed in SEEDS:
        data = root / 'datasets' / f'seed-{seed}'
        tasks = {t.task_id:t for t in [V3RequirementTask.model_validate_json(line) for line in (data/'test_tasks.jsonl').read_text(encoding='utf-8').splitlines()]}
        train = [CompleteSpec.model_validate_json(line) for line in (data/'train_complete_specs.jsonl').read_text(encoding='utf-8').splitlines()]
        model = TypedRelationalModel.fit(train,schema,config)
        engine = ConstraintEngine(schema)
        subset = [row for row in records if row['seed']==seed]
        for row in subset:
            episode = V3EpisodeResult.model_validate(row['episode'])
            task = tasks[row['task_id']]
            if episode.initial_state != task.initial_state:
                raise ValueError('initial state/task binding mismatch')
            state = task.initial_state.public_copy()
            if engine.check_satisfiability(state).satisfiable is not True:
                raise ValueError('initial state not SAT')
            for index,step in enumerate(episode.steps):
                if step.step_index != index or step.field_id not in state.queryable_field_ids() or not schema[step.field_id].queryable:
                    raise ValueError('invalid sequence or eligibility')
                if step.answer.status != FieldStatus.OBSERVED or not _matches_truth(step.answer.value,task.ground_truth[step.field_id],schema[step.field_id]):
                    raise ValueError('oracle answer differs from frozen truth')
                state = state.apply_answer(step.field_id,step.answer)
                if step.satisfiable_after_answer is not True or engine.check_satisfiability(state).satisfiable is not True:
                    raise ValueError('trace feasible flag differs from solver')
                steps += 1
            if state != episode.final_state:
                raise ValueError('trace replay/final state mismatch')
            independent = reviewer_metrics(episode,task,schema,model)
            if set(independent) != set(row['metrics']):
                raise ValueError('metric names mismatch')
            for key,value in independent.items():
                if not math.isclose(value,row['metrics'][key],rel_tol=0,abs_tol=1e-12):
                    raise ValueError(f'metric mismatch {row["key"]} {key}')
                numeric_values += 1
            checked += 1
        print(json.dumps({'verified_seed':seed,'records':len(subset)}),flush=True)
    if sha256(path) != before:
        raise ValueError('records changed during independent verification')
    payload = {'status':'passed','record_count':checked,'step_count':steps,'metric_value_count':numeric_values,
        'records_hash':before,'verifier_source_hash':sha256(Path(__file__)), 'binding':binding,
        'checks':['full key/metadata/hash coverage','task and initial/final state','step eligibility and order',
            'oracle answer equals frozen truth','every real successor SAT','all dual metrics independently recomputed']}
    write_once(root/'independent_verification.json',payload)


if __name__ == '__main__':
    main()
