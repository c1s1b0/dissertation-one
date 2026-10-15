from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from scipy.stats import wilcoxon

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.v2_metrics import v2_episode_metrics
from paper1_active_req.evaluation.statistics import (
    holm_adjust,
    paired_bootstrap_mean_ci,
    paired_rank_biserial,
)
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v2_pivotal_specs import V2RequirementTask
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import EmpiricalAnswerModel
from paper1_active_req.policies.baselines import (
    ConstraintRangeOnlyPolicy,
    CriticalityGreedyPolicy,
    FixedQuestionnairePolicy,
    RandomPolicy,
    UncertaintyAFAPolicy,
)
from paper1_active_req.policies.lookahead import (
    CCLAOneStepPolicy,
    CCLATwoStepPolicy,
    ConditionalAnswerModel,
)
from paper1_active_req.solver.engine import ConstraintEngine

from .episode import EpisodeResult, run_episode
from .runner import _exclusive_output_lock


MODULE_BY_FIELD = {
    "workpiece_mass_kg": "load_space",
    "workpiece_length_mm": "load_space",
    "workpiece_width_mm": "load_space",
    "workpiece_height_mm": "load_space",
    "available_length_mm": "load_space",
    "available_width_mm": "load_space",
    "available_height_mm": "load_space",
    "side_clearance_mm": "load_space",
    "target_throughput_per_hour": "throughput_speed",
    "cycle_time_s": "throughput_speed",
    "travel_distance_mm": "throughput_speed",
    "transfer_time_s": "throughput_speed",
    "conveyor_speed_mm_s": "throughput_speed",
    "positioning_accuracy_mm": "positioning",
    "repeatability_mm": "positioning",
    "workpiece_fragile": "positioning",
    "upstream_output_interface": "interface",
    "station_input_interface": "interface",
    "station_output_interface": "interface",
    "downstream_input_interface": "interface",
}


class V2ExperimentRecord(BaseModel):
    task_id: str
    spec_id: str
    split: str
    missing_rate: float
    mechanism: str
    audit_pivot_selected: bool
    policy_name: str
    config_hash: str
    seed: int
    budget: int
    runtime_s: float
    metrics: dict[str, float]
    episode: EpisodeResult


def _load_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_seed(seed: int, task_id: str, policy_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}:{policy_name}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big")


def _config_hash(*, seed: int, budget: int, lambda_cost: float, lambda_solver: float) -> str:
    payload = json.dumps(
        {
            "seed": seed,
            "budget": budget,
            "lambda_cost": lambda_cost,
            "lambda_solver": lambda_solver,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _validate_v2_paths(*paths: Path) -> None:
    for path in paths:
        normalized = path.resolve().as_posix().lower()
        if "/v2/" not in normalized or any(
            forbidden in normalized
            for forbidden in ("/generated/formal/", "/results/formal/", "/v1/")
        ):
            raise ValueError("v2 paths are required and v1/formal artifacts are forbidden")


def run_v2_policies(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    output_path: Path,
    seed: int,
    budget: int,
    lambda_cost: float,
    lambda_solver: float,
    policy_names: set[str] | None = None,
) -> int:
    _validate_v2_paths(tasks_path, training_specs_path, output_path)
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    tasks = _load_jsonl(tasks_path, V2RequirementTask)
    all_specs = _load_jsonl(training_specs_path, CompleteSpec)
    training_specs = [spec for spec in all_specs if spec.split == "train"]
    if not training_specs:
        raise ValueError("training_specs_path contains no train split")
    conditional_model = ConditionalAnswerModel.fit(
        training_specs=[spec.values for spec in training_specs],
        module_by_field=MODULE_BY_FIELD,
    )
    unconditioned_model = ConditionalAnswerModel.fit(
        training_specs=[spec.values for spec in training_specs],
        module_by_field=MODULE_BY_FIELD,
        use_module_conditioning=False,
    )
    empirical_model = EmpiricalAnswerModel.from_complete_specs(
        values=[spec.values for spec in training_specs],
        max_numeric_outcomes=5,
    )
    config_hash = _config_hash(
        seed=seed,
        budget=budget,
        lambda_cost=lambda_cost,
        lambda_solver=lambda_solver,
    )
    written = 0
    with _exclusive_output_lock(output_path):
        existing: set[tuple[str, str, str]] = set()
        if output_path.exists():
            for record in _load_jsonl(output_path, V2ExperimentRecord):
                existing.add((record.task_id, record.policy_name, record.config_hash))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for task in tasks:
                policies = [
                    RandomPolicy(_stable_seed(seed, task.task_id, "random")),
                    FixedQuestionnairePolicy(),
                    CriticalityGreedyPolicy(),
                    UncertaintyAFAPolicy(empirical_model),
                    ConstraintRangeOnlyPolicy(engine, empirical_model),
                    CCLAOneStepPolicy(
                        engine=engine,
                        answer_model=conditional_model,
                        lambda_cost=lambda_cost,
                        lambda_solver=lambda_solver,
                    ),
                    CCLATwoStepPolicy(
                        engine=engine,
                        answer_model=conditional_model,
                        lambda_cost=lambda_cost,
                        lambda_solver=lambda_solver,
                    ),
                    CCLATwoStepPolicy(
                        engine=engine,
                        answer_model=unconditioned_model,
                        lambda_cost=lambda_cost,
                        lambda_solver=lambda_solver,
                    ),
                ]
                policies[-1].name = "ccla_two_step_unconditioned"
                if policy_names is not None:
                    available = {policy.name for policy in policies}
                    unknown = policy_names - available
                    if unknown:
                        raise ValueError(f"unknown v2 policies: {sorted(unknown)}")
                    policies = [
                        policy for policy in policies if policy.name in policy_names
                    ]
                for policy in policies:
                    key = (task.task_id, policy.name, config_hash)
                    if key in existing:
                        continue
                    started = perf_counter()
                    episode = run_episode(
                        task=task,
                        policy=policy,
                        oracle=SimulatedOracle(task.ground_truth),
                        engine=engine,
                        budget=budget,
                    )
                    runtime_s = perf_counter() - started
                    record = V2ExperimentRecord(
                        task_id=task.task_id,
                        spec_id=task.spec_id,
                        split=task.split,
                        missing_rate=task.missing_rate,
                        mechanism=task.audit.mechanism,
                        audit_pivot_selected=(
                            task.audit.pivot_field_id is not None
                            and any(
                                step.field_id == task.audit.pivot_field_id
                                for step in episode.steps[:budget]
                            )
                        ),
                        policy_name=policy.name,
                        config_hash=config_hash,
                        seed=seed,
                        budget=budget,
                        runtime_s=runtime_s,
                        metrics=v2_episode_metrics(
                            episode, task, schema, conditional_model, engine
                        ),
                        episode=episode,
                    )
                    handle.write(record.model_dump_json() + "\n")
                    handle.flush()
                    existing.add(key)
                    written += 1
    return written


def summarize_v2_results(episodes_path: Path, output_dir: Path) -> None:
    """汇总 v2 的总体与缺失率分层结果，并固定两项主配对比较。"""
    rows = [
        json.loads(line)
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("episodes file is empty")
    frame = pd.DataFrame(
        [
            {
                "task_id": row["task_id"],
                "policy_name": row["policy_name"],
                "missing_rate": row["missing_rate"],
                **row["metrics"],
            }
            for row in rows
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _summary_frame(frame, []).to_csv(output_dir / "metrics.csv", index=False)
    _comparison_frame(frame, []).to_csv(output_dir / "pairwise_tests.csv", index=False)
    _summary_frame(frame, ["missing_rate"]).to_csv(
        output_dir / "stratified_metrics.csv", index=False
    )
    _comparison_frame(frame, ["missing_rate"]).to_csv(
        output_dir / "stratified_pairwise_tests.csv", index=False
    )


def select_v2_lambdas(candidates: list[dict[str, float]]) -> dict[str, object]:
    """从 validation 候选摘要中按预注册规则确定唯一的 λ 组合。"""
    required = {
        "lambda_cost",
        "lambda_solver",
        "crr_at_8",
        "recall_budget_auc",
        "answer_cost",
    }
    if not candidates or any(not required.issubset(candidate) for candidate in candidates):
        raise ValueError("each validation candidate must contain all selection metrics")
    selected = min(
        candidates,
        key=lambda candidate: (
            -float(candidate["crr_at_8"]),
            -float(candidate["recall_budget_auc"]),
            float(candidate["answer_cost"]),
            float(candidate["lambda_cost"]),
            float(candidate["lambda_solver"]),
        ),
    )
    return {
        "selection_split": "validation",
        "selection_rule": "max_crr_at_8_then_auc_then_min_answer_cost_then_lambda_lexicographic",
        "selected_lambdas": {
            "lambda_cost": float(selected["lambda_cost"]),
            "lambda_solver": float(selected["lambda_solver"]),
        },
        "candidate_count": len(candidates),
    }


def run_v2_validation_tuning(
    *,
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    output_dir: Path,
    seed: int,
    budget: int,
) -> dict[str, object]:
    """在 validation 任务上运行冻结的 12 组 λ 候选并写出唯一选择。"""
    _validate_v2_paths(validation_tasks_path, training_specs_path, output_dir)
    raw_tasks = [
        json.loads(line)
        for line in validation_tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_tasks or any(task.get("split") != "validation" for task in raw_tasks):
        raise ValueError("validation tuning requires only validation tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, float]] = []
    for index, (lambda_cost, lambda_solver) in enumerate(
        (cost, solver)
        for cost in (0.0, 0.05, 0.1, 0.2)
        for solver in (0.0, 0.01, 0.05)
    ):
        episodes_path = output_dir / f"candidate-{index:02d}.jsonl"
        run_v2_policies(
            tasks_path=validation_tasks_path,
            training_specs_path=training_specs_path,
            schema_path=schema_path,
            output_path=episodes_path,
            seed=seed,
            budget=budget,
            lambda_cost=lambda_cost,
            lambda_solver=lambda_solver,
            policy_names={"ccla_two_step"},
        )
        records = _load_jsonl(episodes_path, V2ExperimentRecord)
        ccla_records = [
            record for record in records if record.policy_name == "ccla_two_step"
        ]
        candidates.append(
            {
                "lambda_cost": lambda_cost,
                "lambda_solver": lambda_solver,
                "crr_at_8": float(np.mean([r.metrics["crr_at_8"] for r in ccla_records])),
                "recall_budget_auc": float(np.mean([r.metrics["recall_budget_auc"] for r in ccla_records])),
                "answer_cost": float(np.mean([r.metrics["answer_cost"] for r in ccla_records])),
            }
        )
    selected = select_v2_lambdas(candidates)
    payload = {
        **selected,
        "seed": seed,
        "budget": budget,
        "candidates": candidates,
    }
    (output_dir / "selected_lambdas.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return payload


def run_v2_selected_experiment(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    selected_lambdas_path: Path,
    output_path: Path,
    seed: int,
    budget: int,
) -> int:
    """以 validation 选择的 λ 在 test split 上执行唯一的正式运行。"""
    _validate_v2_paths(
        tasks_path, training_specs_path, selected_lambdas_path, output_path
    )
    selected = yaml.safe_load(selected_lambdas_path.read_text(encoding="utf-8"))
    if not isinstance(selected, dict) or selected.get("selection_split") != "validation":
        raise ValueError("selected lambdas must be chosen on validation")
    lambdas = selected.get("selected_lambdas")
    if not isinstance(lambdas, dict) or {"lambda_cost", "lambda_solver"} - set(lambdas):
        raise ValueError("selected lambdas file is malformed")
    raw_tasks = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_tasks or any(task.get("split") != "test" for task in raw_tasks):
        raise ValueError("formal v2 experiment requires only test tasks")
    return run_v2_policies(
        tasks_path=tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        output_path=output_path,
        seed=seed,
        budget=budget,
        lambda_cost=float(lambdas["lambda_cost"]),
        lambda_solver=float(lambdas["lambda_solver"]),
    )


def _summary_frame(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"task_id", "policy_name", *groups}
    ]
    return (
        frame.groupby([*groups, "policy_name"], as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values([*groups, "policy_name"])
    )


def _comparison_frame(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    grouped = [((), frame)] if not groups else frame.groupby(groups, dropna=False)
    comparisons: list[dict[str, float | str]] = []
    for group_key, group_frame in grouped:
        group_values = group_key if isinstance(group_key, tuple) else (group_key,)
        reference = group_frame[group_frame["policy_name"] == "ccla_two_step"][
            ["task_id", "crr_at_8"]
        ].rename(columns={"crr_at_8": "reference_crr"})
        for baseline in ("criticality_greedy", "ccla_one_step"):
            other = group_frame[group_frame["policy_name"] == baseline][
                ["task_id", "crr_at_8"]
            ].rename(columns={"crr_at_8": "baseline_crr"})
            paired = reference.merge(other, on="task_id")
            differences = (paired["reference_crr"] - paired["baseline_crr"]).to_numpy()
            p_value = 1.0 if len(differences) < 10 or np.allclose(differences, 0.0) else float(wilcoxon(differences).pvalue)
            ci_low, ci_high = paired_bootstrap_mean_ci(differences, seed=20260805)
            row: dict[str, float | str] = {
                "baseline": baseline,
                "n": float(len(differences)),
                "mean_difference": float(differences.mean()) if len(differences) else 0.0,
                "effect_size_rank_biserial": paired_rank_biserial(differences),
                "mean_difference_ci_low": ci_low,
                "mean_difference_ci_high": ci_high,
                "p_value": p_value,
            }
            for name, value in zip(groups, group_values, strict=True):
                row[name] = float(value)
            comparisons.append(row)
    if not comparisons:
        return pd.DataFrame()
    adjusted = holm_adjust([float(row["p_value"]) for row in comparisons])
    for row, p_value_holm in zip(comparisons, adjusted, strict=True):
        row["p_value_holm"] = p_value_holm
    return pd.DataFrame(comparisons)
