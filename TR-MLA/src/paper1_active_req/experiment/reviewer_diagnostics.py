"""审稿后探索性校准与约束压力诊断；冻结模型和规则保持只读。"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import z3

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import RelationFamily, load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel
from paper1_active_req.solver.engine import ConstraintEngine, SolverResult


def parent_only_state(spec, target, parents, gate, perturbation, seed):
    """仅公开该关系的父字段/门控字段，所有目标与其他字段严格置空。"""
    if target in parents or target == gate:
        raise ValueError("目标不能同时作为公开父字段")
    values = {key: FieldValue(value=None, status=FieldStatus.HIDDEN, source="reviewer_mask")
              for key in spec.values}
    # 摘要派生种子避免 Python hash 随进程变化，同时不使用目标值。
    digest = hashlib.sha256(f"{seed}:{spec.spec_id}:{target}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    for key in parents:
        value = spec.values[key].value
        value = float(value) * (1 + perturbation * rng.choice((-1, 1)))
        values[key] = FieldValue(value=value, status=FieldStatus.OBSERVED, source="reviewer_public_parent")
    if gate:
        values[gate] = spec.values[gate].model_copy(deep=True)
    return RequirementState(values=values)


def stress_cases(spec, schema):
    """每个完整规格固定清洁例及六个非法输入；修改前后值均可审核。"""
    truth = {key: item.value for key, item in spec.values.items()}
    different_interface = next(x for x in schema["station_input_interface"].categories
                               if x != truth["upstream_output_interface"])
    mutations = {
        "clean": {},
        "interface": {"station_input_interface": different_interface},
        "geometry": {"available_width_mm": truth["workpiece_width_mm"] - 1},
        "cycle_throughput": {"cycle_time_s": 3600 / truth["target_throughput_per_hour"] + 1},
        "load": {"load_capacity_kg": truth["safety_factor"] * truth["workpiece_mass_kg"] - 1},
        "domain": {"working_height_mm": schema["working_height_mm"].upper + 1},
        "mixed": {"station_input_interface": different_interface,
                  "available_width_mm": truth["workpiece_width_mm"] - 1,
                  "load_capacity_kg": truth["safety_factor"] * truth["workpiece_mass_kg"] - 1,
                  "repeatability_mm": truth["repeatability_mm"] * 1.10},
    }
    cases = []
    for kind, changed in mutations.items():
        state = spec.to_requirement_state()
        # 所有完整规格的字段均作为公开观察值。
        state.values = {key: FieldValue(value=item.value, status=FieldStatus.OBSERVED, source="clean_complete")
                        for key, item in state.values.items()}
        for key, value in changed.items():
            state.values[key] = FieldValue(value=value, status=FieldStatus.OBSERVED, source="reviewer_injection")
        cases.append({"source_spec_id": spec.spec_id, "kind": kind, "state": state,
                      "expected_illegal": kind != "clean",
                      "changes": {key: {"before": truth[key], "after": value} for key, value in changed.items()}})
    return cases


def check_with_core(engine, state):
    """直接复用冻结规则构造跟踪求解器，避免 check 的额外全字段边界搜索。"""
    solver = z3.Solver()
    solver.set(timeout=engine.timeout_ms)
    # domain 单独追踪，使越界 core 可读；不改变 domain/rule 的语义。
    for index, constraint in enumerate(engine._domain_constraints()):
        solver.assert_and_track(constraint, z3.Bool(f"domain__{index}"))
    for rule in engine.rules:
        solver.assert_and_track(rule.build(engine.variables), z3.Bool(rule.rule_id))
    for name, expression in engine._observation_constraints(state):
        solver.assert_and_track(expression, z3.Bool(name))
    status = solver.check()
    if status == z3.unknown:
        return SolverResult(satisfiable=None, timed_out=True)
    return SolverResult(satisfiable=status == z3.sat,
                        unsat_core=[str(x) for x in solver.unsat_core()] if status == z3.unsat else [])


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def run_diagnostics(project_root: Path, dataset_dir: Path, output_dir: Path, label: str, spec_limit=None):
    paths = {"train": dataset_dir / "train_complete_specs.jsonl",
             "test": dataset_dir / "test_complete_specs.jsonl", "tasks": dataset_dir / "test_tasks.jsonl",
             "schema": project_root / "configs/fields.yaml", "relations": project_root / "configs/v3_relations.yaml"}
    specs = [CompleteSpec.model_validate(x) for x in _read(paths["test"])]
    if spec_limit is not None:
        specs = specs[:spec_limit]
    train = [CompleteSpec.model_validate(x) for x in _read(paths["train"])]
    schema = load_field_schema(paths["schema"])
    config = load_v3_relation_config(paths["relations"])
    models = {"relation": TypedRelationalModel.fit(train, schema, config),
              "marginal": TypedRelationalModel.fit(train, schema, config, use_relations=False)}
    relations = [x for x in config.relations if x.family != RelationFamily.DISCRETE_CPT]
    rows = []
    for spec in specs:
        for relation in relations:
            target = relation.target_field_id
            truth = float(spec.values[target].value)
            tolerance = config.engineering_tolerances[target]
            for perturbation in (0., .05, .10):
                state = parent_only_state(spec, target, relation.parent_field_ids, relation.gate_field_id,
                                          perturbation, 20261009)
                for name, model in models.items():
                    posterior = model.posterior(state, target)
                    low, high = posterior.interval_low, posterior.interval_high
                    # 保留冻结可靠解析所使用的代表值定义；激活和容差准确性分开报告。
                    predicted = float(min(posterior.branches,
                                          key=lambda b: abs(float(b.value) - (low + high) / 2)).value)
                    width = high - low
                    reliable = posterior.confidence >= config.confidence_threshold or width <= tolerance
                    rows.append({"dataset": label, "spec_id": spec.spec_id, "target": target,
                                 "model": name, "parent_perturbation": perturbation,
                                 "public_parent_values": json.dumps({key: state.values[key].value
                                                                     for key in relation.parent_field_ids}, sort_keys=True),
                                 "truth": truth, "prediction": predicted, "signed_error": predicted - truth,
                                 "absolute_error": abs(predicted - truth), "squared_error": (predicted - truth)**2,
                                 "interval_low": low, "interval_high": high, "coverage95": low <= truth <= high,
                                 "interval_width": width, "tolerance": tolerance, "width_over_tolerance": width / tolerance,
                                 "confidence": posterior.confidence, "reliable": reliable,
                                 "engineering_correct": abs(predicted - truth) <= tolerance,
                                 "exact_correct": abs(predicted - truth) <= 1e-9,
                                 "factor_ids": json.dumps(posterior.factor_ids)})
    summaries = []
    for target in sorted({row["target"] for row in rows}):
        for name in models:
            for perturbation in (0., .05, .10):
                group = [x for x in rows if (x["target"], x["model"], x["parent_perturbation"]) == (target, name, perturbation)]
                summaries.append({"dataset": label, "target": target, "model": name,
                                  "parent_perturbation": perturbation, "n": len(group),
                                  "mae": float(np.mean([x["absolute_error"] for x in group])),
                                  "rmse": float(np.sqrt(np.mean([x["squared_error"] for x in group]))),
                                  **{key: float(np.mean([x[source] for x in group])) for key, source in
                                     (("coverage95", "coverage95"), ("mean_width_over_tolerance", "width_over_tolerance"),
                                      ("reliable_rate", "reliable"), ("engineering_correct_rate", "engineering_correct"),
                                      ("exact_correct_rate", "exact_correct"))}})
    availability = []
    ids = {spec.spec_id for spec in specs}
    for task in _read(paths["tasks"]):
        if task["spec_id"] not in ids:
            continue
        state = RequirementState.model_validate(task["initial_state"])
        for relation in relations:
            parents = relation.parent_field_ids
            known = lambda key: state.values[key].status in {FieldStatus.OBSERVED, FieldStatus.INFERRED} and state.values[key].value is not None
            availability.append({"dataset": label, "task_id": task["task_id"], "spec_id": task["spec_id"],
                                 "missing_rate": task["missing_rate"], "target": relation.target_field_id,
                                 "parent_count": len(parents), "available_parent_count": sum(known(x) for x in parents),
                                 "all_parents_available": all(known(x) for x in parents),
                                 "gate_available": known(relation.gate_field_id) if relation.gate_field_id else True})
    engine = ConstraintEngine(schema)
    stress = []
    for spec in specs:
        for case in stress_cases(spec, schema):
            result = check_with_core(engine, case["state"])
            stress.append({"dataset": label, "source_spec_id": case["source_spec_id"], "kind": case["kind"],
                           "expected_illegal": case["expected_illegal"], "changes": case["changes"],
                           "input_state": case["state"].model_dump(mode="json"),
                           "solver_result": result.model_dump(), "enabled_accept": result.satisfiable is True,
                           "disabled_accept": True, "guard_action": "accept" if result.satisfiable is True else "reject_and_request_confirmation"})
    output_dir.mkdir(parents=True, exist_ok=True)
    _csv(output_dir / f"{label}_continuous.csv", rows)
    _csv(output_dir / f"{label}_continuous_summary.csv", summaries)
    _csv(output_dir / f"{label}_parent_availability.csv", availability)
    availability_summary = []
    for target in sorted({x["target"] for x in availability}):
        for missing_rate in sorted({x["missing_rate"] for x in availability}):
            group = [x for x in availability if (x["target"], x["missing_rate"]) == (target, missing_rate)]
            availability_summary.append({"dataset": label, "target": target, "missing_rate": missing_rate,
                                         "n": len(group),
                                         "all_parents_available_rate": float(np.mean([x["all_parents_available"] for x in group])),
                                         "parents_and_gate_available_rate": float(np.mean([x["all_parents_available"] and x["gate_available"] for x in group]))})
    _csv(output_dir / f"{label}_parent_availability_summary.csv", availability_summary)
    with (output_dir / f"{label}_stress.jsonl").open("w", encoding="utf-8") as handle:
        for row in stress:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stress_summary = {"clean_n": len(specs), "perturbed_n": len(specs)*6}
    for name in ("enabled", "disabled"):
        stress_summary[name] = {
            "false_accept": sum(x["expected_illegal"] and x[f"{name}_accept"] for x in stress),
            "clean_false_reject": sum(not x["expected_illegal"] and not x[f"{name}_accept"] and x["solver_result"]["satisfiable"] is not None for x in stress),
            "unknown": sum(x["solver_result"]["satisfiable"] is None for x in stress) if name == "enabled" else 0}
        stress_summary[name]["false_accept_rate"] = stress_summary[name]["false_accept"] / stress_summary["perturbed_n"]
        stress_summary[name]["clean_false_reject_rate"] = stress_summary[name]["clean_false_reject"] / stress_summary["clean_n"]
    stress_summary["by_kind"] = []
    for kind in sorted({x["kind"] for x in stress}):
        group = [x for x in stress if x["kind"] == kind]
        stress_summary["by_kind"].append({"dataset": label, "kind": kind, "n": len(group),
                                          "enabled_accepted": sum(x["enabled_accept"] for x in group),
                                          "disabled_accepted": len(group),
                                          "unknown": sum(x["solver_result"]["satisfiable"] is None for x in group)})
    _csv(output_dir / f"{label}_stress_summary.csv", stress_summary["by_kind"])
    summary = {"dataset": label, "exploratory": True, "train_n": len(train), "test_n": len(specs),
               "perturbation_seed": 20261009, "confidence_threshold": config.confidence_threshold,
               "continuous": {"row_count": len(rows), "groups": summaries}, "stress": stress_summary,
               "parent_availability": availability_summary,
               "input_hashes": {key: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in paths.items()},
               "source_hashes": {str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), project_root/'src/paper1_active_req/solver/rules.py',
                                              project_root/'src/paper1_active_req/relations/model.py',
                                              project_root/'src/paper1_active_req/relations/continuous.py')},
               "limitations": ["合成基准校准，不构成工业验证", "外层拒绝并要求确认是新增保护，不是原算法纠错", "非法输入标签来自预定义注入，不以求解器结果定义真值", "95%区间来自训练残差分位数，未再校准", "未知求解状态保留，不记为确认误拒绝"]}
    (output_dir / f"{label}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
