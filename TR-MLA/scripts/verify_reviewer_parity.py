"""原冻结轨迹与新增SAT边界执行循环的选择一致性验证。"""
import json
from pathlib import Path
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.experiment.reviewer_runner import SatBoundaryEngine, POLICIES, make_policy, sha256
from paper1_active_req.experiment.v3_episode import run_v3_episode
from paper1_active_req.oracle.simulated import SimulatedOracle


def main():
    project = Path(__file__).resolve().parents[1]
    directory = project / "data/generated/v3"
    schema = load_field_schema(project / "configs/fields.yaml")
    config = load_v3_relation_config(project / "configs/v3_relations.yaml")
    train = [CompleteSpec.model_validate_json(line) for line in (directory / "train_complete_specs.jsonl").read_text(encoding="utf-8").splitlines()]
    tasks = [V3RequirementTask.model_validate_json(line) for line in (directory / "test_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = {}
    for task in tasks:
        key = (task.audit.mechanism,task.missing_rate)
        selected.setdefault(key,task)
    references = {}
    formal = project / "results/v3/formal/episodes.jsonl"
    for line in formal.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        references[(row["task_id"],row["policy_name"])] = row["episode"]
    model = TypedRelationalModel.fit(train,schema,config)
    outcomes = []
    for task in selected.values():
        for name in POLICIES:
            engine = SatBoundaryEngine(schema)
            policy = make_policy(name,task,schema,engine,train,config,model,0)
            episode = run_v3_episode(task=task,policy=policy,oracle=SimulatedOracle(task.ground_truth),engine=engine,budget=8)
            ref = references[(task.task_id,name)]
            observed = [step.field_id for step in episode.steps]
            expected = [step["field_id"] for step in ref["steps"]]
            if observed != expected:
                raise ValueError(f"legacy trajectory mismatch: {task.task_id} {name}: {observed} != {expected}")
            for step,old in zip(episode.steps,ref["steps"],strict=True):
                if step.utility_terms != old["utility_terms"] or step.satisfiable_after_answer != old["satisfiable_after_answer"]:
                    raise ValueError("legacy score/feasibility mismatch")
            outcomes.append({"task_id":task.task_id,"policy":name,"action_count":len(observed),"parity":True})
            print(json.dumps(outcomes[-1]),flush=True)
    target = project / "research_artifacts/reviewer_supplement_20261009/parity.json"
    payload = {"checks":outcomes,"formal_hash":sha256(formal),"status":"passed"}
    with target.open("x",encoding="utf-8") as stream:
        json.dump(payload,stream,indent=2)


if __name__ == "__main__":
    main()
