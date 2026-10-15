from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from pydantic import BaseModel

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.metrics import episode_metrics
from paper1_active_req.generator.complete_specs import (
    CompleteSpec,
    RequirementTask,
)
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import EmpiricalAnswerModel
from paper1_active_req.policies.constraint_state import (
    ConstraintStatePolicy,
    UtilityWeights,
)
from paper1_active_req.solver.engine import ConstraintEngine

from .episode import run_episode
from .runner import _load_jsonl


class WeightSearchRecord(BaseModel):
    weights: UtilityWeights
    mean_crr_at_8: float
    mean_answer_cost: float
    mean_solver_calls: float
    n: int


class SuccessiveHalvingRecord(WeightSearchRecord):
    candidate_id: str
    stage: str
    task_split: str
    task_hash: str
    config_hash: str
    runtime_s: float


def iter_weight_grid(grid: dict[str, list[float]]) -> list[UtilityWeights]:
    names = [
        "domain_reduction",
        "criticality",
        "conflict",
        "graph_degree",
        "answer_cost",
    ]
    missing = [name for name in names if name not in grid]
    if missing:
        raise ValueError(f"weight grid missing keys: {missing}")
    return [
        UtilityWeights.model_validate(dict(zip(names, values, strict=True)))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def sample_weight_candidates(
    grid: dict[str, list[float]],
    default_weights: UtilityWeights,
    *,
    seed: int,
    count: int,
) -> list[UtilityWeights]:
    """Sample a fixed candidate set while retaining the default anchor."""
    candidates = iter_weight_grid(grid)
    default_key = tuple(default_weights.model_dump().values())
    remaining = [
        candidate
        for candidate in candidates
        if tuple(candidate.model_dump().values()) != default_key
    ]
    if len(remaining) == len(candidates):
        raise ValueError("default weights must belong to the configured grid")
    if count < 1 or count > len(candidates):
        raise ValueError("candidate count must be between 1 and grid size")
    if count == 1:
        return [default_weights]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(remaining), size=count - 1, replace=False)
    return [default_weights, *(remaining[int(index)] for index in indices)]


def select_stage_one_tasks(
    tasks: list[RequirementTask], *, seed: int, spec_count: int
) -> list[RequirementTask]:
    """Select complete validation-spec groups for low-fidelity screening."""
    if not tasks or any(task.split != "validation" for task in tasks):
        raise ValueError("stage-one selection requires validation tasks")
    spec_ids = sorted({task.spec_id for task in tasks})
    if spec_count < 1 or spec_count > len(spec_ids):
        raise ValueError("spec_count must be within available validation specs")
    rng = np.random.default_rng(seed)
    selected_ids = sorted(
        spec_ids[int(index)]
        for index in rng.choice(len(spec_ids), size=spec_count, replace=False)
    )
    selected = [task for task in tasks if task.spec_id in selected_ids]
    expected_count = spec_count * 3
    if len(selected) != expected_count:
        raise ValueError(
            "each selected validation spec must have exactly three tasks"
        )
    return sorted(selected, key=lambda task: task.task_id)


def select_best_weight(
    records: list[WeightSearchRecord],
) -> WeightSearchRecord:
    if not records:
        raise ValueError("weight search produced no records")

    def key(record: WeightSearchRecord):
        l1 = sum(record.weights.model_dump().values())
        return (
            -record.mean_crr_at_8,
            record.mean_answer_cost,
            record.mean_solver_calls,
            l1,
            tuple(record.weights.model_dump().values()),
        )

    return min(records, key=key)


def _task_hash(tasks: list[RequirementTask]) -> str:
    payload = "\n".join(sorted(task.task_id for task in tasks)).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _staged_config_hash(
    *,
    config: dict,
    seed: int,
    budget: int,
    candidate_count: int,
    stage_one_spec_count: int,
    finalists: int,
) -> str:
    payload = json.dumps(
        {
            "config": config,
            "seed": seed,
            "budget": budget,
            "candidate_count": candidate_count,
            "stage_one_spec_count": stage_one_spec_count,
            "finalists": finalists,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _evaluate_candidate(
    *,
    candidate_id: str,
    weights: UtilityWeights,
    stage: str,
    tasks: list[RequirementTask],
    task_hash: str,
    config_hash: str,
    schema,
    engine: ConstraintEngine,
    empirical_model: EmpiricalAnswerModel,
    budget: int,
) -> SuccessiveHalvingRecord:
    metrics_rows: list[dict[str, float]] = []
    policy = ConstraintStatePolicy(
        engine=engine,
        empirical_model=empirical_model,
        weights=weights,
    )
    start = perf_counter()
    for task in tasks:
        episode = run_episode(
            task=task,
            policy=policy,
            oracle=SimulatedOracle(task.ground_truth),
            engine=engine,
            budget=budget,
        )
        metrics_rows.append(episode_metrics(episode, task, schema, engine))
    return SuccessiveHalvingRecord(
        candidate_id=candidate_id,
        stage=stage,
        task_split="validation",
        task_hash=task_hash,
        config_hash=config_hash,
        weights=weights,
        mean_crr_at_8=float(
            np.mean([row["crr_at_8"] for row in metrics_rows])
        ),
        mean_answer_cost=float(
            np.mean([row["answer_cost"] for row in metrics_rows])
        ),
        mean_solver_calls=float(
            np.mean([row["solver_calls"] for row in metrics_rows])
        ),
        n=len(metrics_rows),
        runtime_s=perf_counter() - start,
    )


def run_successive_halving_weight_search(
    *,
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    config_path: Path,
    records_path: Path,
    selected_path: Path,
    seed: int,
    budget: int,
    candidate_count: int,
    stage_one_spec_count: int,
    finalists: int,
) -> WeightSearchRecord:
    """Select weights with validation-only random search and halving."""
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    tasks = _load_jsonl(validation_tasks_path, RequirementTask)
    if not tasks or any(task.split != "validation" for task in tasks):
        raise ValueError("weight search requires validation split tasks")
    specs = _load_jsonl(training_specs_path, CompleteSpec)
    training_specs = [spec for spec in specs if spec.split == "train"]
    if not training_specs:
        raise ValueError("weight search requires train split specs")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    default_weights = UtilityWeights.model_validate(config["default_weights"])
    candidates = sample_weight_candidates(
        config["weight_grid"],
        default_weights,
        seed=seed,
        count=candidate_count,
    )
    if finalists < 1 or finalists > len(candidates):
        raise ValueError("finalists must be within candidate count")
    empirical_model = EmpiricalAnswerModel.from_complete_specs(
        values=[spec.values for spec in training_specs],
        max_numeric_outcomes=int(config["max_numeric_outcomes"]),
    )
    stage_one_tasks = select_stage_one_tasks(
        tasks, seed=seed, spec_count=stage_one_spec_count
    )
    stage_tasks = {"stage1": stage_one_tasks, "stage2": tasks}
    task_hashes = {
        stage: _task_hash(stage_task_list)
        for stage, stage_task_list in stage_tasks.items()
    }
    config_hash = _staged_config_hash(
        config=config,
        seed=seed,
        budget=budget,
        candidate_count=candidate_count,
        stage_one_spec_count=stage_one_spec_count,
        finalists=finalists,
    )
    existing: dict[tuple[str, str, str, str], SuccessiveHalvingRecord] = {}
    if records_path.exists():
        for record in _load_jsonl(records_path, SuccessiveHalvingRecord):
            existing[
                (
                    record.candidate_id,
                    record.stage,
                    record.task_hash,
                    record.config_hash,
                )
            ] = record
    records_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_stage(
        stage: str, candidate_ids: list[int]
    ) -> list[SuccessiveHalvingRecord]:
        completed: list[SuccessiveHalvingRecord] = []
        with records_path.open("a", encoding="utf-8") as handle:
            for candidate_index in candidate_ids:
                candidate_id = f"candidate-{candidate_index:03d}"
                key = (
                    candidate_id,
                    stage,
                    task_hashes[stage],
                    config_hash,
                )
                record = existing.get(key)
                if record is None:
                    record = _evaluate_candidate(
                        candidate_id=candidate_id,
                        weights=candidates[candidate_index],
                        stage=stage,
                        tasks=stage_tasks[stage],
                        task_hash=task_hashes[stage],
                        config_hash=config_hash,
                        schema=schema,
                        engine=engine,
                        empirical_model=empirical_model,
                        budget=budget,
                    )
                    handle.write(record.model_dump_json() + "\n")
                    handle.flush()
                    existing[key] = record
                completed.append(record)
        return completed

    stage_one_records = evaluate_stage(
        "stage1", list(range(len(candidates)))
    )
    ranked_stage_one = sorted(
        stage_one_records,
        key=lambda record: (
            -record.mean_crr_at_8,
            record.mean_answer_cost,
            record.mean_solver_calls,
            sum(record.weights.model_dump().values()),
            tuple(record.weights.model_dump().values()),
        ),
    )
    stage_two_indices = [
        int(record.candidate_id.rsplit("-", maxsplit=1)[1])
        for record in ranked_stage_one[:finalists]
    ]
    if 0 not in stage_two_indices:
        stage_two_indices.append(0)
    stage_two_records = evaluate_stage("stage2", stage_two_indices)
    best_record = select_best_weight(stage_two_records)
    selected_candidate_id = next(
        record.candidate_id
        for record in stage_two_records
        if record.weights == best_record.weights
    )
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        yaml.safe_dump(
            {
                "selected_weights": best_record.weights.model_dump(),
                "selection_metric": "critical_requirement_recall_at_8",
                "selection_split": "validation",
                "search_method": "random_search_successive_halving",
                "seed": seed,
                "budget": budget,
                "config_hash": config_hash,
                "candidate_count": len(candidates),
                "stage_one_spec_count": stage_one_spec_count,
                "stage_one_task_hash": task_hashes["stage1"],
                "stage_two_task_hash": task_hashes["stage2"],
                "finalist_candidate_ids": [
                    record.candidate_id for record in stage_two_records
                ],
                "finalist_count": len(stage_two_records),
                "selected_candidate_id": selected_candidate_id,
                "mean_validation_crr_at_8": best_record.mean_crr_at_8,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return best_record


def run_weight_search(
    *,
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    config_path: Path,
    records_path: Path,
    selected_path: Path,
    seed: int,
    budget: int,
) -> WeightSearchRecord:
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    tasks = _load_jsonl(validation_tasks_path, RequirementTask)
    if not tasks or any(task.split != "validation" for task in tasks):
        raise ValueError("weight search requires validation split tasks")
    specs = _load_jsonl(training_specs_path, CompleteSpec)
    training_specs = [spec for spec in specs if spec.split == "train"]
    if not training_specs:
        raise ValueError("weight search requires train split specs")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    empirical_model = EmpiricalAnswerModel.from_complete_specs(
        values=[spec.values for spec in training_specs],
        max_numeric_outcomes=int(config["max_numeric_outcomes"]),
    )

    existing: dict[tuple[float, ...], WeightSearchRecord] = {}
    if records_path.exists():
        for record in _load_jsonl(records_path, WeightSearchRecord):
            key = tuple(record.weights.model_dump().values())
            existing[key] = record
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(existing.values())
    with records_path.open("a", encoding="utf-8") as handle:
        for weights in iter_weight_grid(config["weight_grid"]):
            key = tuple(weights.model_dump().values())
            if key in existing:
                continue
            metrics_rows: list[dict[str, float]] = []
            policy = ConstraintStatePolicy(
                engine=engine,
                empirical_model=empirical_model,
                weights=weights,
            )
            for task in tasks:
                episode = run_episode(
                    task=task,
                    policy=policy,
                    oracle=SimulatedOracle(task.ground_truth),
                    engine=engine,
                    budget=budget,
                )
                metrics_rows.append(
                    episode_metrics(episode, task, schema, engine)
                )
            record = WeightSearchRecord(
                weights=weights,
                mean_crr_at_8=float(
                    np.mean([row["crr_at_8"] for row in metrics_rows])
                ),
                mean_answer_cost=float(
                    np.mean([row["answer_cost"] for row in metrics_rows])
                ),
                mean_solver_calls=float(
                    np.mean([row["solver_calls"] for row in metrics_rows])
                ),
                n=len(metrics_rows),
            )
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            records.append(record)
            existing[key] = record

    best = select_best_weight(records)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        yaml.safe_dump(
            {
                "selected_weights": best.weights.model_dump(),
                "selection_metric": "critical_requirement_recall_at_8",
                "selection_split": "validation",
                "seed": seed,
                "budget": budget,
                "evaluated_weight_combinations": len(records),
                "mean_validation_crr_at_8": best.mean_crr_at_8,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return best
