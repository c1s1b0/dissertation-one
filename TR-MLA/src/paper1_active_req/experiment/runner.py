from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from scipy.stats import wilcoxon

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.metrics import episode_metrics
from paper1_active_req.evaluation.statistics import (
    holm_adjust,
    paired_bootstrap_mean_ci,
    paired_rank_biserial,
)
from paper1_active_req.generator.complete_specs import (
    CompleteSpec,
    RequirementTask,
)
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import EmpiricalAnswerModel
from paper1_active_req.policies.baselines import (
    ConstraintRangeOnlyPolicy,
    CriticalityGreedyPolicy,
    FixedQuestionnairePolicy,
    RandomPolicy,
    UncertaintyAFAPolicy,
)
from paper1_active_req.policies.constraint_state import (
    ConstraintStatePolicy,
    UtilityWeights,
)
from paper1_active_req.solver.engine import ConstraintEngine

from .episode import EpisodeResult, run_episode


class ExperimentRecord(BaseModel):
    task_id: str
    spec_id: str
    split: str
    missing_rate: float
    policy_name: str
    config_hash: str
    seed: int
    budget: int
    runtime_s: float
    metrics: dict[str, float]
    episode: EpisodeResult
    analysis_metadata: dict[str, object] | None = None


def _load_jsonl(path: Path, model_type):
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_seed(seed: int, task_id: str, policy_name: str) -> int:
    payload = f"{seed}:{task_id}:{policy_name}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:4], byteorder="big"
    )


def _config_hash(config: dict, *, seed: int, budget: int) -> str:
    payload = json.dumps(
        {"config": config, "seed": seed, "budget": budget},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


@contextmanager
def _exclusive_output_lock(output_path: Path):
    lock_path = output_path.with_suffix(f"{output_path.suffix}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    try:
        with lock_path.open("x", encoding="utf-8"):
            acquired = True
            yield
    except FileExistsError as error:
        raise FileExistsError(
            f"another writer holds the output lock: {lock_path}"
        ) from error
    finally:
        if acquired and lock_path.exists():
            lock_path.unlink()


def run_all_policies(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    config_path: Path,
    output_path: Path,
    seed: int,
    budget: int,
    selected_weights_path: Path | None = None,
) -> int:
    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    tasks = _load_jsonl(tasks_path, RequirementTask)
    all_specs = _load_jsonl(training_specs_path, CompleteSpec)
    training_specs = [spec for spec in all_specs if spec.split == "train"]
    if not training_specs:
        raise ValueError("training_specs_path contains no train split")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    weights = resolve_utility_weights(config, selected_weights_path)
    resolved_config = {
        **config,
        "resolved_weights": weights.model_dump(),
    }
    config_hash = _config_hash(
        resolved_config, seed=seed, budget=budget
    )
    empirical_model = EmpiricalAnswerModel.from_complete_specs(
        values=[spec.values for spec in training_specs],
        max_numeric_outcomes=int(config["max_numeric_outcomes"]),
    )

    existing: set[tuple[str, str, str]] = set()
    if output_path.exists():
        for record in _load_jsonl(output_path, ExperimentRecord):
            existing.add(
                (record.task_id, record.policy_name, record.config_hash)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            policies = [
                RandomPolicy(
                    _stable_seed(seed, task.task_id, "random")
                ),
                FixedQuestionnairePolicy(),
                CriticalityGreedyPolicy(),
                UncertaintyAFAPolicy(empirical_model),
                ConstraintRangeOnlyPolicy(engine, empirical_model),
                ConstraintStatePolicy(
                    engine=engine,
                    empirical_model=empirical_model,
                    weights=weights,
                ),
            ]
            for policy in policies:
                key = (task.task_id, policy.name, config_hash)
                if key in existing:
                    continue
                oracle = SimulatedOracle(task.ground_truth)
                start = perf_counter()
                episode = run_episode(
                    task=task,
                    policy=policy,
                    oracle=oracle,
                    engine=engine,
                    budget=budget,
                )
                runtime_s = perf_counter() - start
                metrics = episode_metrics(
                    episode, task, schema, engine
                )
                record = ExperimentRecord(
                    task_id=task.task_id,
                    spec_id=task.spec_id,
                    split=task.split,
                    missing_rate=task.missing_rate,
                    policy_name=policy.name,
                    config_hash=config_hash,
                    seed=seed,
                    budget=budget,
                    runtime_s=runtime_s,
                    metrics=metrics,
                    episode=episode,
                )
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                existing.add(key)
                written += 1
    return written


def resolve_utility_weights(
    config: dict,
    selected_weights_path: Path | None,
) -> UtilityWeights:
    if selected_weights_path is None:
        return UtilityWeights.model_validate(config["default_weights"])
    selected = yaml.safe_load(
        selected_weights_path.read_text(encoding="utf-8")
    )
    if selected.get("selection_split") != "validation":
        raise ValueError("selected weights must come from validation split")
    return UtilityWeights.model_validate(selected["selected_weights"])


def run_constraint_state_ablation(
    *,
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    config_path: Path,
    selected_weights_path: Path,
    variant_name: str,
    ablated_term: Literal[
        "domain_reduction", "criticality", "conflict", "answer_cost"
    ],
    output_path: Path,
    seed: int,
    budget: int,
) -> int:
    allowed_terms = {
        "domain_reduction",
        "criticality",
        "conflict",
        "answer_cost",
    }
    if ablated_term not in allowed_terms:
        raise ValueError(f"unsupported ablation term: {ablated_term}")
    if variant_name != f"ablate_{ablated_term}":
        raise ValueError("variant name must match the ablated term")

    schema = load_field_schema(schema_path)
    engine = ConstraintEngine(schema)
    tasks = _load_jsonl(tasks_path, RequirementTask)
    all_specs = _load_jsonl(training_specs_path, CompleteSpec)
    training_specs = [spec for spec in all_specs if spec.split == "train"]
    if not training_specs:
        raise ValueError("training_specs_path contains no train split")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reference_weights = resolve_utility_weights(config, selected_weights_path)
    weights = reference_weights.model_copy(
        update={ablated_term: 0.0}
    )
    selected_payload = yaml.safe_load(
        selected_weights_path.read_text(encoding="utf-8")
    )
    reference_config_hash = _config_hash(
        {**config, "resolved_weights": reference_weights.model_dump()},
        seed=seed,
        budget=budget,
    )
    analysis_metadata: dict[str, object] = {
        "analysis_type": "post_hoc_diagnostic",
        "variant_name": variant_name,
        "ablated_term": ablated_term,
        "reference_config_hash": reference_config_hash,
        "selected_weights_path": str(selected_weights_path),
        "selected_candidate_id": selected_payload.get(
            "selected_candidate_id"
        ),
        "resolved_weights": weights.model_dump(),
    }
    config_hash = _config_hash(
        {**config, "analysis_metadata": analysis_metadata},
        seed=seed,
        budget=budget,
    )
    empirical_model = EmpiricalAnswerModel.from_complete_specs(
        values=[spec.values for spec in training_specs],
        max_numeric_outcomes=int(config["max_numeric_outcomes"]),
    )

    policy_name = f"constraint_state_{variant_name}"
    with _exclusive_output_lock(output_path):
        existing: set[tuple[str, str, str]] = set()
        if output_path.exists():
            for record in _load_jsonl(output_path, ExperimentRecord):
                existing.add(
                    (record.task_id, record.policy_name, record.config_hash)
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with output_path.open("a", encoding="utf-8") as handle:
            for task in tasks:
                key = (task.task_id, policy_name, config_hash)
                if key in existing:
                    continue
                policy = ConstraintStatePolicy(
                    engine=engine,
                    empirical_model=empirical_model,
                    weights=weights,
                )
                policy.name = policy_name
                oracle = SimulatedOracle(task.ground_truth)
                start = perf_counter()
                episode = run_episode(
                    task=task,
                    policy=policy,
                    oracle=oracle,
                    engine=engine,
                    budget=budget,
                )
                runtime_s = perf_counter() - start
                metrics = episode_metrics(episode, task, schema, engine)
                record = ExperimentRecord(
                    task_id=task.task_id,
                    spec_id=task.spec_id,
                    split=task.split,
                    missing_rate=task.missing_rate,
                    policy_name=policy_name,
                    config_hash=config_hash,
                    seed=seed,
                    budget=budget,
                    runtime_s=runtime_s,
                    metrics=metrics,
                    episode=episode,
                    analysis_metadata=analysis_metadata,
                )
                handle.write(record.model_dump_json() + "\n")
                handle.flush()
                existing.add(key)
                written += 1
    return written


def summarize_ablation_results(
    reference_episodes_path: Path,
    ablation_episodes_paths: list[Path],
    output_path: Path,
) -> None:
    reference_records = _load_jsonl(
        reference_episodes_path, ExperimentRecord
    )
    reference_rows = [
        {
            "task_id": record.task_id,
            "crr_at_8": record.metrics["crr_at_8"],
            "recall_budget_auc": record.metrics["recall_budget_auc"],
            "answer_cost": record.metrics["answer_cost"],
        }
        for record in reference_records
        if record.policy_name == "constraint_state_full"
    ]
    if not reference_rows:
        raise ValueError("reference episodes contain no full policy records")
    reference = pd.DataFrame(reference_rows)
    summaries: list[dict[str, float | str]] = []
    for ablation_path in ablation_episodes_paths:
        ablation_records = _load_jsonl(ablation_path, ExperimentRecord)
        if not ablation_records:
            raise ValueError(f"ablation episodes file is empty: {ablation_path}")
        metadata = ablation_records[0].analysis_metadata
        if metadata is None:
            raise ValueError("ablation record is missing analysis metadata")
        if metadata.get("analysis_type") != "post_hoc_diagnostic":
            raise ValueError("ablation record is not post-hoc diagnostic")
        variant_name = metadata.get("variant_name")
        if not isinstance(variant_name, str):
            raise ValueError("ablation record is missing variant name")
        ablation = pd.DataFrame(
            [
                {
                    "task_id": record.task_id,
                    "crr_at_8": record.metrics["crr_at_8"],
                    "recall_budget_auc": record.metrics[
                        "recall_budget_auc"
                    ],
                    "answer_cost": record.metrics["answer_cost"],
                }
                for record in ablation_records
            ]
        )
        paired = reference.merge(
            ablation,
            on="task_id",
            suffixes=("_full", "_ablation"),
        )
        if len(paired) != len(reference) or len(paired) != len(ablation):
            raise ValueError(
                f"ablation tasks do not match reference: {variant_name}"
            )
        crr_differences = (
            paired["crr_at_8_full"] - paired["crr_at_8_ablation"]
        ).to_numpy()
        auc_differences = (
            paired["recall_budget_auc_full"]
            - paired["recall_budget_auc_ablation"]
        ).to_numpy()
        cost_differences = (
            paired["answer_cost_full"]
            - paired["answer_cost_ablation"]
        ).to_numpy()
        crr_ci = paired_bootstrap_mean_ci(crr_differences, seed=20260731)
        auc_ci = paired_bootstrap_mean_ci(auc_differences, seed=20260731)
        cost_ci = paired_bootstrap_mean_ci(cost_differences, seed=20260731)
        summaries.append(
            {
                "analysis_type": "post_hoc_diagnostic",
                "variant_name": variant_name,
                "ablated_term": str(metadata.get("ablated_term")),
                "n": float(len(paired)),
                "mean_crr_difference_full_minus_ablation": float(
                    crr_differences.mean()
                ),
                "crr_effect_size_rank_biserial": paired_rank_biserial(
                    crr_differences
                ),
                "crr_difference_ci_low": crr_ci[0],
                "crr_difference_ci_high": crr_ci[1],
                "mean_auc_difference_full_minus_ablation": float(
                    auc_differences.mean()
                ),
                "auc_effect_size_rank_biserial": paired_rank_biserial(
                    auc_differences
                ),
                "auc_difference_ci_low": auc_ci[0],
                "auc_difference_ci_high": auc_ci[1],
                "mean_answer_cost_difference_full_minus_ablation": float(
                    cost_differences.mean()
                ),
                "answer_cost_effect_size_rank_biserial": (
                    paired_rank_biserial(cost_differences)
                ),
                "answer_cost_difference_ci_low": cost_ci[0],
                "answer_cost_difference_ci_high": cost_ci[1],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output_path, index=False)


def _summarize_frame(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    analysis_type: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {"task_id", "policy_name", *group_columns}
    ]
    summary = (
        frame.groupby(
            [*group_columns, "policy_name"], as_index=False
        )[metric_columns]
        .mean(numeric_only=True)
        .sort_values([*group_columns, "policy_name"])
    )
    counts = (
        frame.groupby([*group_columns, "policy_name"])
        .size()
        .rename("n")
        .reset_index()
    )
    summary = summary.merge(
        counts, on=[*group_columns, "policy_name"]
    )

    comparisons: list[dict[str, float | str]] = []
    grouping = [((), frame)]
    if group_columns:
        grouping = frame.groupby(group_columns, dropna=False)
    for group_key, group_frame in grouping:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        full = group_frame[
            group_frame["policy_name"] == "constraint_state_full"
        ][["task_id", "crr_at_8"]].rename(
            columns={"crr_at_8": "full_crr"}
        )
        group_comparisons: list[dict[str, float | str]] = []
        baselines = sorted(
            set(group_frame["policy_name"]) - {"constraint_state_full"}
        )
        for baseline in baselines:
            other = group_frame[group_frame["policy_name"] == baseline][
                ["task_id", "crr_at_8"]
            ].rename(columns={"crr_at_8": "baseline_crr"})
            paired = full.merge(other, on="task_id")
            differences = (
                paired["full_crr"] - paired["baseline_crr"]
            ).to_numpy()
            if len(differences) < 10 or np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(differences, alternative="two-sided").pvalue
                )
            comparison: dict[str, float | str] = {
                "baseline": baseline,
                "n": float(len(differences)),
                "mean_difference": float(
                    differences.mean() if len(differences) else 0.0
                ),
                "effect_size_rank_biserial": paired_rank_biserial(
                    differences
                ),
                "mean_difference_ci_low": paired_bootstrap_mean_ci(
                    differences, seed=20260731
                )[0],
                "mean_difference_ci_high": paired_bootstrap_mean_ci(
                    differences, seed=20260731
                )[1],
                "p_value": p_value,
            }
            for column, value in zip(group_columns, keys, strict=True):
                comparison[column] = float(value)
            group_comparisons.append(comparison)
        adjusted = holm_adjust(
            [float(item["p_value"]) for item in group_comparisons]
        )
        for item, value in zip(
            group_comparisons, adjusted, strict=True
        ):
            item["p_value_holm"] = value
        comparisons.extend(group_comparisons)

    comparison_frame = pd.DataFrame(comparisons)
    if analysis_type is not None:
        summary["analysis_type"] = analysis_type
        comparison_frame["analysis_type"] = analysis_type
    return summary, comparison_frame


def summarize_results(
    episodes_path: Path,
    output_dir: Path,
    *,
    stratify_by_missing_rate: bool = False,
) -> None:
    records = _load_jsonl(episodes_path, ExperimentRecord)
    if not records:
        raise ValueError("episodes file is empty")
    rows = [
        {
            "task_id": record.task_id,
            "policy_name": record.policy_name,
            "missing_rate": record.missing_rate,
            "runtime_s": record.runtime_s,
            **record.metrics,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    summary, comparisons = _summarize_frame(
        frame.drop(columns="missing_rate"), group_columns=[]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    comparisons.to_csv(output_dir / "pairwise_tests.csv", index=False)
    if stratify_by_missing_rate:
        stratified_summary, stratified_comparisons = _summarize_frame(
            frame,
            group_columns=["missing_rate"],
            analysis_type="post_hoc_diagnostic",
        )
        stratified_summary.to_csv(
            output_dir / "stratified_metrics.csv", index=False
        )
        stratified_comparisons.to_csv(
            output_dir / "stratified_pairwise_tests.csv", index=False
        )
