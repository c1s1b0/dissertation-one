"""仅预算终点使用关系myopic，其他七问与原TR轨迹相同。"""
from pathlib import Path
import json
import pandas as pd
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.experiment.reviewer_runner import SEEDS, SatBoundaryEngine, make_policy, sha256, write_once
from paper1_active_req.experiment.reviewer_terminal import finish_episode
from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / "research_artifacts/reviewer_supplement_20261009"
    target = root / "terminal_diagnostic"
    protocol = target / "protocol.json"
    settings = {"analysis_label":"post_review_exploratory_finite_budget_diagnostic",
        "rule":"retain original seven-query prefix; final query relational-myopic; lambda=0; B=8",
        "seeds":list(SEEDS),"raw_tr_mla_preserved":True,
        "motivation":"frozen TR scorer ignores remaining budget in two-step term",
        "source_hashes":{str(p.relative_to(project)):sha256(p) for p in [Path(__file__),project / "src/paper1_active_req/experiment/reviewer_terminal.py"]}}
    if not protocol.exists():
        write_once(protocol,settings)
    elif json.loads(protocol.read_text(encoding="utf-8")) != settings:
        raise ValueError("terminal diagnostic protocol mismatch")
    outcome_path = root / "execution_outcome.json"
    if not outcome_path.exists():
        print("PROTOCOL_SAVED_WAITING_FOR_MAIN_COMPLETION")
        return
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    record_path = root / "records/episodes.jsonl"
    if outcome["status"] != "complete" or outcome["records_hash"] != sha256(record_path):
        raise ValueError("main execution not verified")
    records = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    schema = load_field_schema(project / "configs/fields.yaml")
    config = load_v3_relation_config(project / "configs/v3_relations.yaml")
    rows = []
    output = target / "episodes.jsonl"
    if output.exists():
        raise FileExistsError("terminal records already exist")
    with output.open("x",encoding="utf-8") as stream:
        for seed in SEEDS:
            data = root / "datasets" / f"seed-{seed}"
            train = [CompleteSpec.model_validate_json(line) for line in (data / "train_complete_specs.jsonl").read_text(encoding="utf-8").splitlines()]
            tasks = {t.task_id:t for t in [V3RequirementTask.model_validate_json(line) for line in (data / "test_tasks.jsonl").read_text(encoding="utf-8").splitlines()]}
            model = TypedRelationalModel.fit(train,schema,config)
            original_rows = [r for r in records if r['seed']==seed and r['policy_name']=='tr_mla' and r['lambda_cost']==0]
            if len(original_rows) != 180:
                raise ValueError("terminal roster incomplete")
            for source in original_rows:
                task = tasks[source['task_id']]
                original = V3EpisodeResult.model_validate(source['episode'])
                engine = SatBoundaryEngine(schema)
                policy = make_policy('typed_relational_myopic',task,schema,engine,train,config,model,0)
                revised = finish_episode(original,task,policy,engine)
                if revised.steps[:7] != original.steps[:7]:
                    raise ValueError("prefix changed")
                metrics = reviewer_metrics(revised,task,schema,model)
                row = {'seed':seed,'task_id':task.task_id,'spec_id':task.spec_id,'missing_rate':task.missing_rate,
                    'changed_final_query':[s.field_id for s in revised.steps]!=[s.field_id for s in original.steps],
                    **metrics,**{'delta_'+k:metrics[k]-source['metrics'][k] for k in ('dcrr_at_8','rcrr_at_8','answer_cost')}}
                rows.append(row)
                stream.write(json.dumps({'summary':row,'episode':revised.model_dump(mode='json')},ensure_ascii=False)+'\n')
            print(json.dumps({'seed':seed,'terminal_records':len(original_rows)}),flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(target / "metrics.csv",index=False)
    primary = frame[frame.missing_rate.isin([.5,.7])].groupby(['seed','spec_id'],as_index=False).mean(numeric_only=True)
    primary.to_csv(target / "spec_m50_m70.csv",index=False)
    primary.groupby('seed').mean(numeric_only=True).to_csv(target / "primary_means.csv")
    write_once(target / "outcome.json",{'status':'complete','n':len(rows),'main_records_hash':sha256(record_path),
        'records_hash':sha256(output),'protocol_hash':sha256(protocol),'interpretation':'local repair diagnostic; not proof of general lookahead superiority'})


if __name__ == '__main__':
    main()
