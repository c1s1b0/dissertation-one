"""冷策略/冷引擎决策计时与固定24字段规模诊断。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.experiment.reviewer_efficiency import InstrumentedEngine, measure_decision, scale_mask
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.policies.baselines import CriticalityGreedyPolicy
from paper1_active_req.policies.marginal_lookahead import MarginalLookaheadNoRelationsPolicy, TRMLAPolicy, TypedRelationalMyopicPolicy
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("criticality_greedy", "marginal_lookahead_no_relations", "typed_relational_myopic", "tr_mla")


def read_models(path, cls):
    return [cls.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def worker(job, queue):
    """进程隔离仪器的临时Z3方法包装；每次新模型、策略和引擎。"""
    try:
        schema = load_field_schema(ROOT / "configs/fields.yaml")
        config = load_v3_relation_config(ROOT / "configs/v3_relations.yaml")
        train = read_models(ROOT / "data/generated/v3/train_complete_specs.jsonl", CompleteSpec)
        task = V3RequirementTask.model_validate(job["task"])
        state = task.initial_state.public_copy()
        if job["group"] == "scale":
            state, schema = scale_mask(state, schema, job["requested_candidates"])
        engine = InstrumentedEngine(schema)
        context = PolicyContext(field_schema=schema, budget=8, step_index=0)
        if job["group"] == "full_boundary":
            start = perf_counter()
            boundary = engine.check(state)
            queue.put({"status": "ok", "full_boundary_seconds": perf_counter() - start,
                       "sat_calls": engine.sat_calls, "unknown_checks": engine.unknown_checks,
                       "boundary_satisfiable": boundary.satisfiable})
            return
        if job["group"] == "fit":
            start = perf_counter()
            TypedRelationalModel.fit(train, schema, config, use_relations=job["use_relations"])
            queue.put({"status": "ok", "fit_seconds": perf_counter() - start})
            return
        name = job["policy"]
        fit_start = perf_counter()
        if name == "criticality_greedy":
            policy = CriticalityGreedyPolicy()
        elif name == "marginal_lookahead_no_relations":
            policy = MarginalLookaheadNoRelationsPolicy.from_training_specs(
                initial_state=state, training_specs=train, schema=schema, relation_config=config,
                engine=engine, lambda_cost=0.0)
        else:
            model = TypedRelationalModel.fit(train, schema, config)
            cls = TRMLAPolicy if name == "tr_mla" else TypedRelationalMyopicPolicy
            policy = cls(initial_state=state, model=model, engine=engine, lambda_cost=0.0)
        construction_seconds = perf_counter() - fit_start
        start = perf_counter()
        boundary = engine.check_satisfiability(state)
        boundary_seconds = perf_counter() - start
        # 初始边界自身使用纯SAT缓存；反事实不与该初始状态相同。
        row = measure_decision(policy, engine, state, boundary, context)
        row.update(status="ok", preparation_seconds=construction_seconds,
                   boundary_sat_seconds=boundary_seconds, boundary_satisfiable=boundary.satisfiable,
                   public_hidden_count=len(state.queryable_field_ids()))
        queue.put(row)
    except Exception as exc:
        queue.put({"status": "error", "error": repr(exc)})


def hardware():
    info = {"os": platform.platform(), "cpu": platform.processor(), "python": sys.version,
            "logical_cpus": mp.cpu_count(), "packages": {}}
    try:
        command = "$c=Get-CimInstance Win32_Processor; $s=Get-CimInstance Win32_ComputerSystem; @{cpu=$c.Name;ram_bytes=$s.TotalPhysicalMemory}|ConvertTo-Json -Compress"
        result = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True, check=True)
        info.update(json.loads(result.stdout))
    except Exception as exc:
        info["hardware_query_error"] = repr(exc)
    for name in ("numpy", "pandas", "scipy", "pydantic", "z3-solver", "PyYAML"):
        info["packages"][name] = importlib.metadata.version(name)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "research_artifacts/reviewer_supplement_20261009/efficiency")
    parser.add_argument("--worker-timeout", type=float, default=60.0)
    parser.add_argument("--skip-full-boundary", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw_replicates.jsonl"
    if raw_path.exists():
        raise FileExistsError("原始效率记录已存在，请使用新的输出目录；禁止覆盖")
    tasks = read_models(ROOT / "data/generated/v3/test_tasks.jsonl", V3RequirementTask)
    # 每机制ID最小规格，按全部三种缺失副本测量，不按性能挑选。
    selected = {mechanism: min(t.spec_id for t in tasks if t.audit.mechanism == mechanism)
                for mechanism in sorted({t.audit.mechanism for t in tasks})}
    chosen = sorted((t for t in tasks if selected[t.audit.mechanism] == t.spec_id), key=lambda t: t.task_id)
    sources = [ROOT / "data/generated/v3/train_complete_specs.jsonl", ROOT / "data/generated/v3/test_tasks.jsonl",
               ROOT / "configs/fields.yaml", ROOT / "configs/v3_relations.yaml", Path(__file__),
               ROOT / "src/paper1_active_req/experiment/reviewer_efficiency.py",
               ROOT / "src/paper1_active_req/policies/marginal_lookahead.py", ROOT / "src/paper1_active_req/solver/engine.py"]
    protocol = {"hardware": hardware(), "replicates": 3, "selected_specs": selected,
                "policies": POLICIES, "candidate_sizes": [2, 4, 8, 12], "worker_timeout_seconds": args.worker_timeout,
                "timed_window": "policy.select only; fitting, initial feasibility and process startup excluded",
                "scale_protocol": "artificial all-hidden state; lexicographically first n schema-queryable fields; fixed 24 fields",
                "input_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = []
    for rep in range(3):
        for use_relations in (False, True):
            jobs.append(dict(group="fit", replicate=rep, use_relations=use_relations, task=chosen[0].model_dump(mode="json")))
        for task in chosen:
            for policy in POLICIES:
                jobs.append(dict(group="missing_rate", replicate=rep, policy=policy, task=task.model_dump(mode="json")))
        for task in [t for t in chosen if t.missing_rate == 0.7]:
            for n in (2, 4, 8, 12):
                for policy in POLICIES:
                    jobs.append(dict(group="scale", replicate=rep, requested_candidates=n, policy=policy, task=task.model_dump(mode="json")))
            if not args.skip_full_boundary:
                jobs.append(dict(group="full_boundary", replicate=rep, task=task.model_dump(mode="json")))
    records = []
    ctx = mp.get_context("spawn")
    with raw_path.open("x", encoding="utf-8") as stream:
        for index, job in enumerate(jobs):
            queue = ctx.Queue()
            process = ctx.Process(target=worker, args=(job, queue))
            process.start()
            start = perf_counter()
            process.join(args.worker_timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                result = {"status": "worker_timeout", "decision_seconds": None,
                          "error": "整个工作进程超过上限；纯select未完成，不能作为完成计时"}
            elif not queue.empty():
                result = queue.get(timeout=2)
            else:
                result = {"status": "worker_crash", "exitcode": process.exitcode}
            queue.close()
            task = job["task"]
            row = {k: v for k, v in job.items() if k != "task"}
            row.update(task_id=task["task_id"], spec_id=task["spec_id"],
                       mechanism=task["audit"]["mechanism"], missing_rate=task["missing_rate"],
                       worker_wall_seconds=perf_counter()-start, **result)
            records.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index+1}/{len(jobs)} {row['group']} {row.get('policy', '')} {row['status']} {row.get('decision_seconds', '')}", flush=True)
    columns = sorted(set().union(*(r.keys() for r in records)))
    with (args.output / "raw_replicates.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    summaries = []
    groups = {}
    for row in records:
        key = (row["group"], row.get("policy"), row["spec_id"],
               row["missing_rate"] if row["group"] == "missing_rate" else None,
               row.get("requested_candidates"), row.get("use_relations"))
        groups.setdefault(key, []).append(row)
    for key, rows in groups.items():
        summary = dict(zip(("group", "policy", "spec_id", "missing_rate", "requested_candidates", "use_relations"), key))
        summary.update(total=len(rows), completed=sum(r["status"] == "ok" for r in rows),
                       failed=sum(r["status"] != "ok" for r in rows))
        for column in ("decision_seconds", "fit_seconds", "full_boundary_seconds", "actual_eligible", "counterfactual_branch_requests", "sat_calls", "unknown_checks"):
            values = [r[column] for r in rows if r["status"] == "ok" and r.get(column) is not None]
            if values:
                summary[column + "_median"] = float(np.median(values))
                summary[column + "_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        summaries.append(summary)
    (args.output / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "测量说明.md").write_text(
        "# 效率补充测量说明\n\n本实验为审稿后探索性测量。三个固定规格为每机制测试ID最小者，分别测m30/m50/m70，三次重复。每条工作进程重新创建策略、拟合模型及约束引擎，不跨重复复用后验或引擎缓存。计时仅包围policy.select；训练拟合、引擎构造、初始可行性与进程启动均排除。实际反事实SAT缓存允许在一次决策内部发挥作用。\n\n"
        "四策略忽略solver_result，因此初始准备使用原始check_satisfiability；完整check包含全字段边界搜索，另外独立计时，不并入decision_seconds。SAT计数是实际Z3 Solver.check次数，反事实请求计数包括缓存命中；unknown完整记录。包装仅在独立串行工作进程内生效，不适用于多线程共享Z3。\n\n"
        "候选规模采用固定24字段模型的人工全隐藏状态，按字段ID字典序只开放2/4/8/12个查询入口；没有把隐藏真值公开。全部18条约束与关系保留。公开缺失状态经过可靠解析剔除后实际eligible可能少于目标n，原始行逐一报告。这是局部候选规模诊断，不是任意大规模机械图谱、更多字段或真实工程部署验证。\n\n"
        "超过整个工作进程上限时保留worker_timeout行与null纯决策时间；不以零值、截止时间或未完成耗时冒充完成测量。summary报告每单元三重复的median/IQR与成功/失败数；IQR只是重复抖动，不是泛化置信区间。同一机器可能同时运行其他补充实验，操作系统调度和仪器开销包含在select窗口。\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()

