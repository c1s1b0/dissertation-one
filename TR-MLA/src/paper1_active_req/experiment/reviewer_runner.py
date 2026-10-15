"""审稿后独立补充运行；不改写v3正式记录或冻结模块。"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.v3_episode import run_v3_episode
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import (
    V3RequirementTask, _MECHANISMS, _audit_for, _generate_specs_and_tasks,
)
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.baselines import CriticalityGreedyPolicy
from paper1_active_req.policies.marginal_lookahead import (
    TRMLAPolicy, TypedRelationalMyopicPolicy, MarginalLookaheadNoRelationsPolicy,
)
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine

SEEDS = (20261011, 20261013, 20261017)
POLICIES = ("criticality_greedy", "marginal_lookahead_no_relations", "typed_relational_myopic", "tr_mla")
LAMBDAS = (0.0, .05, .10, .20)
ROOT_NAME = "reviewer_supplement_20261009"
_CONTEXT = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_root(root: Path) -> Path:
    resolved = root.resolve()
    if resolved.name != ROOT_NAME or resolved.parent.name != "research_artifacts":
        raise ValueError("supplement requires isolated research_artifacts root")
    return resolved


def write_once(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def bind_protocol(root: Path, settings: dict, inputs: dict[str, Path]) -> str:
    root = validate_root(root)
    payload = {"settings": settings, "input_hashes": {key: sha256(path) for key, path in sorted(inputs.items())}}
    digest = json_hash(payload)
    target = root / "protocol.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("protocol_hash") != digest or previous.get("binding") != payload:
            raise ValueError("protocol binding changed; refusing to resume")
    else:
        write_once(target, {"protocol_hash": digest, "binding": payload, "analysis_label": "post_review_exploratory"})
    return digest


def read_checkpoint(path: Path, protocol_hash: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"empty checkpoint line {number}")
        row = json.loads(line)
        if row.get("protocol_hash") != protocol_hash:
            raise ValueError("checkpoint protocol mismatch")
        key = row["key"]
        if key in rows:
            raise ValueError("duplicate checkpoint key")
        if row.get("status") not in {"success", "failed"}:
            raise ValueError("invalid checkpoint status")
        rows[key] = row
    return rows


def protocol_settings() -> dict:
    return {"seeds": list(SEEDS), "policies": list(POLICIES), "budget": 8,
        "split_counts": {"train": 180, "validation": 60, "test": 60},
        "missing_rates": [.3, .5, .7], "pivot_per_mechanism": 15,
        "lambda_grid": list(LAMBDAS), "primary_lambda": 0.0,
        "cost_subset": "per mechanism sorted first 6 specification IDs, m70; two relational policies",
        "metrics": "DCRR direct correct; RCRR common train relation model direct or reliable exact inference",
        "statistical_unit": "within-specification mean m50/m70; bootstrap 10000",
        "formal_pooling": "prohibited", "boundary_engine": "existing SAT check; policies ignore domains"}


def protected_inputs(project: Path) -> dict[str, Path]:
    inputs = {"formal_episodes": project / "results/v3/formal/episodes.jsonl",
        "formal_freeze": project / "results/v3/freeze_manifest.json",
        "schema": project / "configs/fields.yaml", "relation_config": project / "configs/v3_relations.yaml",
        "runner_source": Path(__file__),
        "metrics_source": project / "src/paper1_active_req/evaluation/reviewer_metrics.py"}
    freeze = json.loads(inputs["formal_freeze"].read_text(encoding="utf-8"))
    for source in freeze["source_hashes"]:
        path = Path(source)
        inputs["legacy_source:" + path.relative_to(project).as_posix()] = path
    return inputs


def prepare_datasets(root: Path, project: Path, protocol_hash: str) -> None:
    root = validate_root(root)
    schema = load_field_schema(project / "configs/fields.yaml")
    config = load_v3_relation_config(project / "configs/v3_relations.yaml")
    for seed in SEEDS:
        target = root / "datasets" / f"seed-{seed}"
        lock = target / "manifest.json"
        if lock.exists():
            manifest = json.loads(lock.read_text(encoding="utf-8"))
            if manifest["protocol_hash"] != protocol_hash or any(sha256(target / name) != digest for name, digest in manifest["files"].items()):
                raise ValueError("dataset binding/hash mismatch")
            continue
        if target.exists():
            raise ValueError("incomplete dataset directory; review before resuming")
        target.mkdir(parents=True)
        engine = ConstraintEngine(schema)
        files = {}
        for split, count in (("train",180),("validation",60),("test",60)):
            if split == "test":
                definitions = [(f"supp-{seed}-test-{mechanism}-{index:03d}", _audit_for(mechanism, pivot=index <= 15))
                    for mechanism in _MECHANISMS for index in range(1,21)]
            else:
                definitions = [(f"supp-{seed}-{split}-{index:04d}", _audit_for(_MECHANISMS[index % 3], pivot=True))
                    for index in range(1,count+1)]
            specs, tasks = _generate_specs_and_tasks(master_seed=seed, split=split, spec_definitions=definitions,
                schema=schema, engine=engine, relation_config=config)
            for name, items in ((f"{split}_complete_specs.jsonl", specs),(f"{split}_tasks.jsonl",tasks)):
                with (target / name).open("x", encoding="utf-8", newline="\n") as stream:
                    for item in items:
                        stream.write(item.model_dump_json() + "\n")
                files[name] = sha256(target / name)
            print(json.dumps({"event":"dataset_split","seed":seed,"split":split,"specs":len(specs),"tasks":len(tasks)}), flush=True)
            engine = ConstraintEngine(schema)
        write_once(lock, {"protocol_hash": protocol_hash, "seed": seed, "files": files, "split_counts": {"train":180,"validation":60,"test":60}})


def job_plan(tasks: list[V3RequirementTask], seed: int) -> list[dict]:
    subsets = set()
    for mechanism in _MECHANISMS:
        spec_ids = sorted({t.spec_id for t in tasks if t.audit.mechanism == mechanism})[:6]
        subsets.update(spec_ids)
    jobs = []
    for task in tasks:
        settings = [(name, 0.0, "multiseed") for name in POLICIES]
        if task.spec_id in subsets and task.missing_rate == .7:
            settings += [(name, lam, "cost") for lam in LAMBDAS[1:] for name in ("typed_relational_myopic", "tr_mla")]
        for name, lam, group in settings:
            jobs.append({"key": f"{seed}|{task.task_id}|{name}|{lam:.2f}", "seed":seed,
                "task_id":task.task_id,"policy_name":name,"lambda_cost":lam,"group":group,
                "cost_subset": task.spec_id in subsets and task.missing_rate == .7})
    return jobs


class SatBoundaryEngine(ConstraintEngine):
    """四选定策略不使用domains；边界用原纯SAT检查，选择规则未变。"""
    def check(self, state):
        return self.check_satisfiability(state)


def make_policy(name, task, schema, engine, train, config, model, lam):
    if name == "criticality_greedy":
        return CriticalityGreedyPolicy()
    if name == "marginal_lookahead_no_relations":
        return MarginalLookaheadNoRelationsPolicy.from_training_specs(initial_state=task.initial_state,
            training_specs=train, schema=schema, relation_config=config, engine=engine, lambda_cost=lam)
    cls = {"tr_mla": TRMLAPolicy, "typed_relational_myopic": TypedRelationalMyopicPolicy}[name]
    return cls(initial_state=task.initial_state, model=model, engine=engine, lambda_cost=lam)


def _init_worker(root_path: str, project_path: str):
    root, project = Path(root_path), Path(project_path)
    schema = load_field_schema(project / "configs/fields.yaml")
    config = load_v3_relation_config(project / "configs/v3_relations.yaml")
    data = {}
    for seed in SEEDS:
        directory = root / "datasets" / f"seed-{seed}"
        train = [CompleteSpec.model_validate_json(line) for line in (directory / "train_complete_specs.jsonl").read_text(encoding="utf-8").splitlines()]
        tasks = {task.task_id: task for task in (V3RequirementTask.model_validate_json(line) for line in (directory / "test_tasks.jsonl").read_text(encoding="utf-8").splitlines())}
        if any(spec.split != "train" for spec in train):
            raise ValueError("non-training fit input")
        data[seed] = (train,tasks,TypedRelationalModel.fit(train,schema,config))
    _CONTEXT.update(schema=schema,config=config,data=data)


def _run_job(job: dict) -> dict:
    from paper1_active_req.evaluation.reviewer_metrics import reviewer_metrics
    started = time.perf_counter()
    try:
        train,tasks,model = _CONTEXT["data"][job["seed"]]
        task = tasks[job["task_id"]]
        engine = SatBoundaryEngine(_CONTEXT["schema"])
        policy = make_policy(job["policy_name"],task,_CONTEXT["schema"],engine,train,_CONTEXT["config"],model,job["lambda_cost"])
        episode = run_v3_episode(task=task,policy=policy,oracle=SimulatedOracle(task.ground_truth),engine=engine,budget=8)
        if any(step.satisfiable_after_answer is not True for step in episode.steps):
            raise ValueError("generated clean task did not remain SAT")
        metrics = reviewer_metrics(episode,task,_CONTEXT["schema"],model)
        return {**job,"status":"success","spec_id":task.spec_id,"mechanism":task.audit.mechanism,
            "pivot":task.audit.pivot_field_id is not None,"missing_rate":task.missing_rate,
            "episode":episode.model_dump(mode="json"),"metrics":metrics,
            "wall_seconds":time.perf_counter()-started,"engine_cache":engine.cache_info()}
    except Exception:
        return {**job,"status":"failed","error":traceback.format_exc(),"wall_seconds":time.perf_counter()-started}


def run_supplement(root: Path, project: Path, workers: int = 6) -> dict:
    root = validate_root(root)
    digest = bind_protocol(root,protocol_settings(),protected_inputs(project))
    prepare_datasets(root,project,digest)
    path = root / "records" / "episodes.jsonl"
    previous = read_checkpoint(path,digest)
    jobs = []
    for seed in SEEDS:
        tasks = [V3RequirementTask.model_validate_json(line) for line in (root / "datasets" / f"seed-{seed}" / "test_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        jobs.extend(job_plan(tasks,seed))
    expected = {j["key"] for j in jobs}
    if not set(previous).issubset(expected):
        raise ValueError("unknown checkpoint key")
    if any(row["status"] != "success" for row in previous.values()):
        raise ValueError("failed checkpoint retained; investigate before adding new attempts")
    pending = [job for job in jobs if job["key"] not in previous]
    path.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"event":"start","total":len(jobs),"completed":len(previous),"pending":len(pending),"workers":workers}),flush=True)
    with path.open("a",encoding="utf-8",newline="\n") as stream, ProcessPoolExecutor(max_workers=workers,
            initializer=_init_worker,initargs=(str(root),str(project))) as pool:
        futures = {pool.submit(_run_job,job):job for job in pending}
        for future in as_completed(futures):
            row = future.result()
            row["protocol_hash"] = digest
            stream.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
            stream.flush()
            previous[row["key"]] = row
            if len(previous) % 25 == 0 or row["status"] != "success":
                print(json.dumps({"event":"checkpoint","completed":len(previous),"total":len(jobs),"last_status":row["status"],"last_seconds":row["wall_seconds"]}),flush=True)
            if row["status"] != "success":
                for item in futures:
                    item.cancel()
                raise RuntimeError("supplement failed; checkpoint retained")
    if set(previous) != expected:
        raise ValueError("incomplete supplement")
    bind_protocol(root,protocol_settings(),protected_inputs(project))
    payload = {"status":"complete","protocol_hash":digest,"record_count":len(previous),"expected_count":len(jobs),
        "records_hash":sha256(path),"seeds":list(SEEDS),"analysis_label":"post_review_exploratory"}
    outcome = root / "execution_outcome.json"
    if outcome.exists():
        if json.loads(outcome.read_text(encoding="utf-8")) != payload:
            raise ValueError("completed outcome differs")
    else:
        write_once(outcome,payload)
    return payload
