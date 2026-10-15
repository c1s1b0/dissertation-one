"""与正式 v3 路径隔离的独立鲁棒性研究协议。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import rankdata, wilcoxon

from paper1_active_req.domain.models import FieldDefinition, FieldKind, FieldStatus, FieldValue
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.v3_protocol import frozen_critical_field_ids
import paper1_active_req.generator.v3_relational_specs as v3_generator
import paper1_active_req.experiment.v3_runner as v3_runner
from paper1_active_req.experiment.v3_runner import (
    V3ExperimentRecord,
    V3_POLICY_NAMES,
    run_v3_policies,
    select_v3_lambda_candidate,
)
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.generator.v3_relational_specs import V3RequirementTask
from paper1_active_req.relations.config import load_v3_relation_config
from paper1_active_req.relations.model import TypedRelationalModel


ROBUSTNESS_SEEDS = (20260813, 20260817, 20260823)
_SIDECAR_PARENT_NAME = "research_artifacts"
_SIDECAR_ROOT_NAME = "v3_robustness_20260810"
_ROBUSTNESS_TRAIN_SPEC_COUNT = 180
_ROBUSTNESS_VALIDATION_SPEC_COUNT = 60
_ROBUSTNESS_TEST_SPECS_PER_MECHANISM = 20
_ROBUSTNESS_PIVOT_SPECS_PER_MECHANISM = 15
_ROBUSTNESS_MISSING_RATES = (0.3, 0.5, 0.7)
_ROBUSTNESS_BUDGET = 8
_ROBUSTNESS_LAMBDA_CANDIDATES = (0.0, 0.05, 0.1, 0.2)
_ROBUSTNESS_BOOTSTRAP_REPLICATES = 10_000
_ANALYSIS_LABEL = "independent_robustness_exploratory"


class V3RobustnessRunManifest(BaseModel):
    """单个独立种子的不可覆盖运行摘要。"""

    analysis_label: str = _ANALYSIS_LABEL
    status: str = "succeeded"
    seed: int
    budget: int
    selected_lambda: float
    record_count: int
    policy_names: list[str]
    artifact_hashes: dict[str, str] = Field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _write_text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)


def _write_json_once(path: Path, payload: dict[str, object]) -> None:
    _write_text_once(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _write_jsonl_once(path: Path, values: list[BaseModel]) -> None:
    _write_text_once(path, "\n".join(value.model_dump_json() for value in values) + "\n")


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"robustness JSON object is required: {path.name}")
    return raw


def _load_jsonl(path: Path, model: type[BaseModel]) -> list[Any]:
    return [model.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_sidecar_root(root: Path) -> Path:
    resolved = root.resolve()
    if (
        resolved.name != _SIDECAR_ROOT_NAME
        or resolved.parent.name != _SIDECAR_PARENT_NAME
    ):
        raise ValueError("robustness output must use the isolated sidecar root")
    return resolved


def _validate_seed(seed: int) -> None:
    if seed not in ROBUSTNESS_SEEDS:
        raise ValueError("robustness seed is not predeclared")


def prepare_v3_robustness_run(*, root: Path, seed: int) -> Path:
    """为预先固定的独立种子创建不可复用的 sidecar 运行目录。"""
    _validate_seed(seed)
    sidecar_root = _validate_sidecar_root(root)
    run_dir = sidecar_root / "runs" / f"seed-{seed}"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    return run_dir


def write_v3_robustness_protocol(*, root: Path) -> dict[str, object]:
    """在任何种子运行前写入不可覆盖的独立稳健性协议。"""
    sidecar_root = _validate_sidecar_root(root)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    test_specification = {
        "total": _ROBUSTNESS_TEST_SPECS_PER_MECHANISM * len(v3_generator._MECHANISMS),
        "per_mechanism": _ROBUSTNESS_TEST_SPECS_PER_MECHANISM,
        "pivot_per_mechanism": _ROBUSTNESS_PIVOT_SPECS_PER_MECHANISM,
        "control_per_mechanism": (
            _ROBUSTNESS_TEST_SPECS_PER_MECHANISM - _ROBUSTNESS_PIVOT_SPECS_PER_MECHANISM
        ),
    }
    if test_specification != {
        "total": 60,
        "per_mechanism": 20,
        "pivot_per_mechanism": 15,
        "control_per_mechanism": 5,
    }:
        raise ValueError("robustness test specification must remain fixed at 60/20/15/5")
    payload: dict[str, object] = {
        "analysis_label": _ANALYSIS_LABEL,
        "formal_pooling": "prohibited",
        "seeds": list(ROBUSTNESS_SEEDS),
        "development_specification": {
            "train": _ROBUSTNESS_TRAIN_SPEC_COUNT,
            "validation": _ROBUSTNESS_VALIDATION_SPEC_COUNT,
        },
        "test_specification": test_specification,
        "mechanisms": list(v3_generator._MECHANISMS),
        "missing_rates": list(_ROBUSTNESS_MISSING_RATES),
        "budget": _ROBUSTNESS_BUDGET,
        "policy_names": list(V3_POLICY_NAMES),
        "lambda_candidates": list(_ROBUSTNESS_LAMBDA_CANDIDATES),
        "statistics": {
            "primary": "within-spec mean of m50 and m70",
            "high_pressure": "m70 and pivot=true",
            "bootstrap_replicates": _ROBUSTNESS_BOOTSTRAP_REPLICATES,
            "bootstrap_seed_rule": "seed for primary; seed+1 for high-pressure stratum",
        },
    }
    _write_json_once(sidecar_root / "protocol.json", payload)
    return payload


def write_protected_input_snapshot(
    *,
    root: Path,
    protected_paths: dict[str, Path],
) -> dict[str, object]:
    """在任何 sidecar 运行前固定受保护 formal 输入的 SHA-256 快照。"""
    sidecar_root = _validate_sidecar_root(root)
    if not protected_paths or any(not label for label in protected_paths):
        raise ValueError("protected input snapshot requires named paths")
    hashes: dict[str, str] = {}
    for label, path in protected_paths.items():
        if not path.is_file():
            raise ValueError(f"protected input is unavailable: {label}")
        hashes[label] = _sha256(path)
    sidecar_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "analysis_label": _ANALYSIS_LABEL,
        "protected_inputs": hashes,
    }
    _write_json_once(sidecar_root / "pre_execution_protected_input_hashes.json", payload)
    return payload


def _write_development_dataset(
    *,
    run_dir: Path,
    seed: int,
    schema_path: Path,
    relation_config_path: Path,
) -> tuple[Path, Path]:
    schema, engine, relation_config = v3_generator._load_generation_context(
        schema_path, relation_config_path
    )
    development_dir = run_dir / "development"
    development_dir.mkdir()
    split_specs: dict[str, list[CompleteSpec]] = {}
    split_tasks: dict[str, list[V3RequirementTask]] = {}
    for split, count in (
        ("train", _ROBUSTNESS_TRAIN_SPEC_COUNT),
        ("validation", _ROBUSTNESS_VALIDATION_SPEC_COUNT),
    ):
        definitions = [
            (
                f"v3r-{seed}-{split}-{index:04d}",
                v3_generator._audit_for(
                    v3_generator._MECHANISMS[index % len(v3_generator._MECHANISMS)],
                    pivot=True,
                ),
            )
            for index in range(1, count + 1)
        ]
        specs, tasks = v3_generator._generate_specs_and_tasks(
            master_seed=seed,
            split=split,
            spec_definitions=definitions,
            schema=schema,
            engine=engine,
            relation_config=relation_config,
        )
        split_specs[split] = specs
        split_tasks[split] = tasks
        _write_jsonl_once(development_dir / f"{split}_complete_specs.jsonl", specs)
        _write_jsonl_once(development_dir / f"{split}_tasks.jsonl", tasks)
    manifest_path = development_dir / "development_manifest.json"
    _write_json_once(
        manifest_path,
        {
            "analysis_label": "independent_robustness_exploratory",
            "seed": seed,
            "split_spec_counts": {split: len(items) for split, items in split_specs.items()},
            "split_task_counts": {split: len(items) for split, items in split_tasks.items()},
            "schema_path": str(schema_path.resolve()),
            "relation_config_path": str(relation_config_path.resolve()),
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
        },
    )
    return development_dir / "train_complete_specs.jsonl", development_dir / "validation_tasks.jsonl"


def _tune_on_validation(
    *,
    run_dir: Path,
    seed: int,
    budget: int,
    validation_tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
) -> tuple[float, Path]:
    tuning_dir = run_dir / "tuning"
    tuning_dir.mkdir()
    candidates: list[dict[str, object]] = []
    for lambda_cost in _ROBUSTNESS_LAMBDA_CANDIDATES:
        record_path = tuning_dir / f"candidate-lambda-{lambda_cost:.2f}.jsonl"
        run_v3_policies(
            tasks_path=validation_tasks_path,
            training_specs_path=training_specs_path,
            schema_path=schema_path,
            relation_config_path=relation_config_path,
            output_path=record_path,
            seed=seed,
            budget=budget,
            lambda_cost=lambda_cost,
            policy_names={"tr_mla"},
        )
        records = [
            record.model_dump(mode="json")
            for record in _load_jsonl(record_path, V3ExperimentRecord)
        ]
        candidates.append(
            {
                "lambda_cost": lambda_cost,
                "policy_names": ["tr_mla"],
                "records_path": record_path.name,
                "records": records,
            }
        )
    selected = select_v3_lambda_candidate(candidates)
    selection_path = tuning_dir / "selected_lambda.json"
    _write_json_once(
        selection_path,
        {
            "analysis_label": "independent_robustness_exploratory",
            "selection_split": "validation",
            "seed": seed,
            "budget": budget,
            "selected_lambda": {"lambda_cost": float(selected["lambda_cost"])},
            "selection_metrics": selected["selection_metrics"],
            "validation_tasks_hash": _sha256(validation_tasks_path),
            "training_specs_hash": _sha256(training_specs_path),
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
            "candidate_records_hash": _json_hash(candidates),
            "candidates": candidates,
        },
    )
    return float(selected["lambda_cost"]), selection_path


def _freeze_run(
    *,
    run_dir: Path,
    seed: int,
    budget: int,
    selected_lambda: float,
    training_specs_path: Path,
    validation_tasks_path: Path,
    selection_path: Path,
    schema_path: Path,
    relation_config_path: Path,
) -> Path:
    source_paths = (
        Path(__file__),
        Path(v3_generator.__file__).resolve(),
        Path(v3_runner.__file__).resolve(),
    )
    source_hashes = {
        str(path): _sha256(path)
        for path in source_paths
        if path.is_file()
    }
    freeze_path = run_dir / "freeze_manifest.json"
    _write_json_once(
        freeze_path,
        {
            "analysis_label": "independent_robustness_exploratory",
            "status": "frozen",
            "seed": seed,
            "budget": budget,
            "selected_lambda": selected_lambda,
            "training_specs_hash": _sha256(training_specs_path),
            "validation_tasks_hash": _sha256(validation_tasks_path),
            "selection_hash": _sha256(selection_path),
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
            "source_hashes": source_hashes,
        },
    )
    return freeze_path


def _release_test_dataset(
    *,
    run_dir: Path,
    seed: int,
    freeze_path: Path,
    schema_path: Path,
    relation_config_path: Path,
) -> Path:
    frozen = _load_json(freeze_path)
    if frozen.get("status") != "frozen" or frozen.get("seed") != seed:
        raise ValueError("robustness test release requires matching frozen manifest")
    schema, engine, relation_config = v3_generator._load_generation_context(
        schema_path, relation_config_path
    )
    release_dir = run_dir / "release"
    release_dir.mkdir()
    definitions = [
        (
            f"v3r-{seed}-test-{mechanism}-{index:03d}",
            v3_generator._audit_for(
                mechanism,
                pivot=index <= _ROBUSTNESS_PIVOT_SPECS_PER_MECHANISM,
            ),
        )
        for mechanism in v3_generator._MECHANISMS
        for index in range(1, _ROBUSTNESS_TEST_SPECS_PER_MECHANISM + 1)
    ]
    specs, tasks = v3_generator._generate_specs_and_tasks(
        master_seed=seed,
        split="test",
        spec_definitions=definitions,
        schema=schema,
        engine=engine,
        relation_config=relation_config,
    )
    specs_path = release_dir / "test_complete_specs.jsonl"
    tasks_path = release_dir / "test_tasks.jsonl"
    _write_jsonl_once(specs_path, specs)
    _write_jsonl_once(tasks_path, tasks)
    _write_json_once(
        release_dir / "test_release_manifest.json",
        {
            "analysis_label": "independent_robustness_exploratory",
            "seed": seed,
            "freeze_manifest_hash": _sha256(freeze_path),
            "test_specs_hash": _sha256(specs_path),
            "test_tasks_hash": _sha256(tasks_path),
            "test_spec_ids": [spec.spec_id for spec in specs],
            "test_task_ids": [task.task_id for task in tasks],
            "schema_hash": _sha256(schema_path),
            "relation_config_hash": _sha256(relation_config_path),
        },
    )
    return tasks_path


def _validate_records(records_path: Path, expected_task_count: int) -> list[V3ExperimentRecord]:
    records = _load_jsonl(records_path, V3ExperimentRecord)
    expected_count = expected_task_count * len(V3_POLICY_NAMES)
    if len(records) != expected_count:
        raise ValueError("robustness record count does not match task/policy roster")
    keys = [(record.spec_id, record.missing_rate, record.policy_name) for record in records]
    if len(set(keys)) != expected_count:
        raise ValueError("robustness records contain duplicate task/policy keys")
    if {record.policy_name for record in records} != set(V3_POLICY_NAMES):
        raise ValueError("robustness records do not contain the ten-policy roster")
    if any(record.split != "test" for record in records):
        raise ValueError("robustness records must contain test tasks only")
    return records


def _records_frame(records: list[V3ExperimentRecord]) -> pd.DataFrame:
    """将已验证的侧车 records 转为只含公开实验字段的分析表。"""
    rows = [
        {
            "spec_id": record.spec_id,
            "policy_name": record.policy_name,
            "missing_rate": record.missing_rate,
            "mechanism": record.mechanism,
            "pivot": record.pivot,
            **record.metrics,
        }
        for record in records
    ]
    frame = pd.DataFrame(rows)
    required = {
        "spec_id",
        "policy_name",
        "missing_rate",
        "mechanism",
        "pivot",
        "crr_at_8",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"robustness records are missing analysis fields: {sorted(missing)}")
    return frame


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    return float(np.sum(np.sign(nonzero) * ranks) / np.sum(ranks))


def _paired_effect(
    frame: pd.DataFrame,
    *,
    seed: int,
    scope: str,
) -> dict[str, object]:
    """计算侧车中的配对效应；结果仅用于独立探索性稳健性描述。"""
    subset = frame.loc[
        frame["policy_name"].isin(("tr_mla", "criticality_greedy")),
        ["spec_id", "policy_name", "crr_at_8"],
    ]
    if subset.duplicated(["spec_id", "policy_name"]).any():
        raise ValueError(f"{scope} has duplicate spec/policy records")
    paired = subset.pivot(index="spec_id", columns="policy_name", values="crr_at_8")
    required_policies = {"tr_mla", "criticality_greedy"}
    if not required_policies.issubset(paired.columns):
        raise ValueError(f"{scope} is missing the primary policy pair")
    paired = paired[["tr_mla", "criticality_greedy"]]
    if paired.isna().any().any() or paired.empty:
        raise ValueError(f"{scope} requires complete primary policy pairs")
    differences = (
        paired["tr_mla"] - paired["criticality_greedy"]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        differences.size,
        size=(_ROBUSTNESS_BOOTSTRAP_REPLICATES, differences.size),
    )
    bootstrap_means = differences[samples].mean(axis=1)
    if np.all(differences == 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(differences, alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    ci_low, ci_high = np.quantile(bootstrap_means, (0.025, 0.975))
    return {
        "analysis_label": _ANALYSIS_LABEL,
        "scope": scope,
        "comparison": "tr_mla_minus_criticality_greedy",
        "metric": "crr_at_8",
        "n": int(differences.size),
        "mean_difference": float(differences.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": _rank_biserial(differences),
        "bootstrap_seed": seed,
        "bootstrap_replicates": _ROBUSTNESS_BOOTSTRAP_REPLICATES,
    }


def _aggregate_primary_records(frame: pd.DataFrame) -> pd.DataFrame:
    """固定 m50/m70 为 spec 内均值，并在聚合前验证完整率格。"""
    primary = frame.loc[frame["missing_rate"].isin((0.5, 0.7))].copy()
    if primary.empty:
        raise ValueError("robustness primary analysis requires m50 and m70 records")
    cells = primary.groupby(["spec_id", "policy_name", "missing_rate"]).size()
    if not cells.eq(1).all():
        raise ValueError("robustness primary analysis has duplicate rate cells")
    rates = primary.groupby(["spec_id", "policy_name"])["missing_rate"].agg(set)
    if not rates.map(lambda values: values == {0.5, 0.7}).all():
        raise ValueError("robustness primary analysis requires m50/m70 for every policy")
    numeric_columns = [
        column
        for column in primary.columns
        if column not in {"spec_id", "policy_name", "missing_rate", "mechanism", "pivot"}
        and pd.api.types.is_numeric_dtype(primary[column])
    ]
    aggregated = (
        primary.groupby(["spec_id", "policy_name", "mechanism", "pivot"], as_index=False)[
            numeric_columns
        ]
        .mean()
        .sort_values(["spec_id", "policy_name"], kind="stable")
        .reset_index(drop=True)
    )
    return aggregated


def _matches_truth(predicted: object, truth: FieldValue, definition: FieldDefinition) -> bool:
    if truth.value is None:
        return predicted is None
    if definition.kind in {FieldKind.CONTINUOUS, FieldKind.INTEGER}:
        return (
            isinstance(predicted, (int, float))
            and isinstance(truth.value, (int, float))
            and bool(np.isclose(float(predicted), float(truth.value), rtol=0.0, atol=1e-9))
        )
    return predicted == truth.value


def _summarize_tr_mla_inference(
    *,
    records: list[V3ExperimentRecord],
    tasks_path: Path,
    training_specs_path: Path,
    schema_path: Path,
    relation_config_path: Path,
) -> dict[str, object]:
    """从 episode 逐条重建 TR-MLA 可靠推断，绝不以总体指标倒推计数。"""
    tr_mla_records = [record for record in records if record.policy_name == "tr_mla"]
    if not tr_mla_records:
        raise ValueError("robustness inference summary requires tr_mla records")
    tasks = _load_jsonl(tasks_path, V3RequirementTask)
    task_by_id = {task.task_id: task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("robustness release tasks contain duplicate task IDs")
    schema = load_field_schema(schema_path)
    relation_config = load_v3_relation_config(relation_config_path)
    model = TypedRelationalModel.fit(
        _load_jsonl(training_specs_path, CompleteSpec),
        schema,
        relation_config,
        use_relations=True,
    )
    inferred_total = 0
    inferred_correct = 0
    continuous_reliable_inference_count = 0
    episode_with_inference_count = 0
    zero_filled_precisions: list[float] = []
    for record in tr_mla_records:
        task = task_by_id.get(record.task_id)
        if task is None:
            raise ValueError(f"robustness record references unknown task: {record.task_id}")
        if record.episode.task_id != record.task_id or record.episode.policy_name != "tr_mla":
            raise ValueError(f"robustness episode binding is invalid: {record.task_id}")
        reliably_inferred = model.reliably_resolved(record.episode.final_state.public_copy())
        episode_total = 0
        episode_correct = 0
        for field_id in frozen_critical_field_ids(task.initial_state, schema):
            definition = schema[field_id]
            final_value = record.episode.final_state.values[field_id]
            truth = task.ground_truth[field_id]
            direct_correct = (
                final_value.status == FieldStatus.OBSERVED
                and _matches_truth(final_value.value, truth, definition)
            )
            if direct_correct or field_id not in reliably_inferred:
                continue
            episode_total += 1
            correct = _matches_truth(reliably_inferred[field_id], truth, definition)
            episode_correct += int(correct)
            if definition.kind == FieldKind.CONTINUOUS:
                continuous_reliable_inference_count += 1
        metric_precision = record.metrics.get("inference_precision")
        episode_precision = float(episode_correct / episode_total) if episode_total else 0.0
        if metric_precision is None or not np.isclose(
            episode_precision, float(metric_precision), rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "reconstructed TR-MLA inference precision does not match the robustness record"
            )
        inferred_total += episode_total
        inferred_correct += episode_correct
        episode_with_inference_count += int(episode_total > 0)
        zero_filled_precisions.append(episode_precision)
    return {
        "analysis_label": _ANALYSIS_LABEL,
        "policy_name": "tr_mla",
        "tr_mla_record_count": len(tr_mla_records),
        "zero_filled_episode_precision_mean": float(np.mean(zero_filled_precisions)),
        "episode_with_inference_count": episode_with_inference_count,
        "episode_with_inference_proportion": float(
            episode_with_inference_count / len(tr_mla_records)
        ),
        "inferred_total": inferred_total,
        "inferred_correct": inferred_correct,
        "micro_precision": float(inferred_correct / inferred_total) if inferred_total else 0.0,
        "continuous_reliable_inference_count": continuous_reliable_inference_count,
        "metric_rule": "abs_tol=1e-9 for continuous/integer; equality otherwise",
    }


def summarize_v3_robustness_run(*, root: Path, seed: int) -> dict[str, object]:
    """汇总单个独立种子，明确标记为不可与正式检验合并的探索性结果。"""
    _validate_seed(seed)
    sidecar_root = _validate_sidecar_root(root)
    run_dir = sidecar_root / "runs" / f"seed-{seed}"
    outcome_path = run_dir / "robustness-outcome.json"
    records_path = run_dir / "records" / "episodes.jsonl"
    tasks_path = run_dir / "release" / "test_tasks.jsonl"
    outcome = _load_json(outcome_path)
    if outcome.get("status") != "succeeded" or outcome.get("seed") != seed:
        raise ValueError("robustness summary requires a matching succeeded outcome")
    if outcome.get("records_hash") != _sha256(records_path):
        raise ValueError("robustness records hash does not match outcome")
    records = _validate_records(
        records_path,
        expected_task_count=len(_load_jsonl(tasks_path, V3RequirementTask)),
    )
    if outcome.get("record_count") != len(records):
        raise ValueError("robustness outcome record count does not match records")
    frame = _records_frame(records)
    primary = _aggregate_primary_records(frame)
    high_pressure = frame.loc[(frame["missing_rate"] == 0.7) & frame["pivot"]].copy()
    development_manifest = _load_json(run_dir / "development" / "development_manifest.json")
    schema_path = Path(str(development_manifest.get("schema_path", "")))
    relation_config_path = Path(str(development_manifest.get("relation_config_path", "")))
    if (
        not schema_path.is_file()
        or not relation_config_path.is_file()
        or development_manifest.get("schema_hash") != _sha256(schema_path)
        or development_manifest.get("relation_config_hash") != _sha256(relation_config_path)
    ):
        raise ValueError("robustness summary source schema or relation configuration is unavailable")
    inference = _summarize_tr_mla_inference(
        records=records,
        tasks_path=tasks_path,
        training_specs_path=run_dir / "development" / "train_complete_specs.jsonl",
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    summary_dir = run_dir / "summary"
    if summary_dir.exists():
        raise FileExistsError("refusing to overwrite robustness summary")
    summary_dir.mkdir()
    primary.to_csv(summary_dir / "primary_m50_m70.csv", index=False)
    high_pressure.to_csv(summary_dir / "high_pressure_m70_pivot.csv", index=False)
    frame.groupby(["policy_name", "mechanism"], as_index=False)["crr_at_8"].mean().to_csv(
        summary_dir / "mechanism_crr_at_8.csv", index=False
    )
    pd.DataFrame([inference]).to_csv(summary_dir / "inference.csv", index=False)
    payload: dict[str, object] = {
        "analysis_label": _ANALYSIS_LABEL,
        "seed": seed,
        "primary": _paired_effect(primary, seed=seed, scope="m50_m70_within_spec_mean"),
        "high_pressure_m70_pivot": _paired_effect(
            high_pressure, seed=seed + 1, scope="m70_pivot_true"
        ),
        "inference": inference,
        "formal_pooling": "prohibited",
    }
    _write_json_once(summary_dir / "robustness_effect.json", payload)
    return payload


def summarize_v3_robustness_cross_seed(*, root: Path) -> dict[str, object]:
    """并排汇总预声明三种子效应，不计算也不输出任何跨种子 pooled p value。"""
    sidecar_root = _validate_sidecar_root(root)
    protocol = _load_json(sidecar_root / "protocol.json")
    if (
        protocol.get("analysis_label") != _ANALYSIS_LABEL
        or protocol.get("formal_pooling") != "prohibited"
        or protocol.get("seeds") != list(ROBUSTNESS_SEEDS)
    ):
        raise ValueError("robustness cross-seed summary requires the immutable protocol")
    rows: list[dict[str, object]] = []
    for seed in ROBUSTNESS_SEEDS:
        effect = _load_json(
            sidecar_root / "runs" / f"seed-{seed}" / "summary" / "robustness_effect.json"
        )
        if (
            effect.get("analysis_label") != _ANALYSIS_LABEL
            or effect.get("formal_pooling") != "prohibited"
            or effect.get("seed") != seed
        ):
            raise ValueError(f"robustness effect has an invalid protocol binding for seed {seed}")
        for stratum in ("primary", "high_pressure_m70_pivot"):
            item = effect.get(stratum)
            if not isinstance(item, dict):
                raise ValueError(f"robustness effect is missing {stratum} for seed {seed}")
            try:
                rows.append(
                    {
                        "analysis_label": _ANALYSIS_LABEL,
                        "seed": seed,
                        "stratum": stratum,
                        "n": int(item["n"]),
                        "mean_difference": float(item["mean_difference"]),
                        "ci_low": float(item["ci_low"]),
                        "ci_high": float(item["ci_high"]),
                        "comparison": "tr_mla_minus_criticality_greedy",
                    }
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"robustness effect has invalid {stratum} fields for seed {seed}"
                ) from error
    output_dir = sidecar_root / "summary"
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite cross-seed robustness summary")
    output_dir.mkdir()
    pd.DataFrame(rows).to_csv(output_dir / "cross_seed_effects.csv", index=False)
    payload: dict[str, object] = {
        "analysis_label": _ANALYSIS_LABEL,
        "formal_pooling": "prohibited",
        "seeds": list(ROBUSTNESS_SEEDS),
        "effect_count": len(rows),
        "cross_seed_inference": "not computed",
    }
    _write_json_once(output_dir / "cross_seed_summary.json", payload)
    return payload


def run_v3_robustness_run(
    *,
    root: Path,
    seed: int,
    schema_path: Path,
    relation_config_path: Path,
    budget: int = _ROBUSTNESS_BUDGET,
) -> V3RobustnessRunManifest:
    """运行一个独立种子，且不访问或写入 v3 formal 工件。"""
    _validate_seed(seed)
    if budget != _ROBUSTNESS_BUDGET:
        raise ValueError("robustness budget must remain fixed at 8")
    if _ROBUSTNESS_LAMBDA_CANDIDATES != (0.0, 0.05, 0.1, 0.2):
        raise ValueError("robustness lambda grid must remain frozen")
    run_dir = prepare_v3_robustness_run(root=root, seed=seed)
    training_specs_path, validation_tasks_path = _write_development_dataset(
        run_dir=run_dir,
        seed=seed,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    selected_lambda, selection_path = _tune_on_validation(
        run_dir=run_dir,
        seed=seed,
        budget=budget,
        validation_tasks_path=validation_tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    freeze_path = _freeze_run(
        run_dir=run_dir,
        seed=seed,
        budget=budget,
        selected_lambda=selected_lambda,
        training_specs_path=training_specs_path,
        validation_tasks_path=validation_tasks_path,
        selection_path=selection_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    test_tasks_path = _release_test_dataset(
        run_dir=run_dir,
        seed=seed,
        freeze_path=freeze_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
    )
    records_path = run_dir / "records" / "episodes.jsonl"
    written = run_v3_policies(
        tasks_path=test_tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        output_path=records_path,
        seed=seed,
        budget=budget,
        lambda_cost=selected_lambda,
    )
    records = _validate_records(records_path, expected_task_count=len(_load_jsonl(test_tasks_path, V3RequirementTask)))
    if written != len(records):
        raise ValueError("robustness writer count does not match records")
    outcome_path = run_dir / "robustness-outcome.json"
    _write_json_once(
        outcome_path,
        {
            "analysis_label": "independent_robustness_exploratory",
            "status": "succeeded",
            "seed": seed,
            "record_count": len(records),
            "records_hash": _sha256(records_path),
            "policy_names": list(V3_POLICY_NAMES),
        },
    )
    artifact_hashes = {
        "development_manifest": _sha256(run_dir / "development" / "development_manifest.json"),
        "selection": _sha256(selection_path),
        "freeze_manifest": _sha256(freeze_path),
        "release_manifest": _sha256(run_dir / "release" / "test_release_manifest.json"),
        "records": _sha256(records_path),
        "outcome": _sha256(outcome_path),
    }
    manifest = V3RobustnessRunManifest(
        seed=seed,
        budget=budget,
        selected_lambda=selected_lambda,
        record_count=len(records),
        policy_names=list(V3_POLICY_NAMES),
        artifact_hashes=artifact_hashes,
    )
    _write_json_once(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
    return manifest


def resume_v3_robustness_run(
    *,
    root: Path,
    seed: int,
    schema_path: Path,
    relation_config_path: Path,
    budget: int = _ROBUSTNESS_BUDGET,
) -> V3RobustnessRunManifest:
    """从已冻结、已释放且未产生轨迹的 sidecar 种子继续测试阶段。"""
    _validate_seed(seed)
    if budget != _ROBUSTNESS_BUDGET:
        raise ValueError("robustness budget must remain fixed at 8")
    sidecar_root = _validate_sidecar_root(root)
    run_dir = sidecar_root / "runs" / f"seed-{seed}"
    if not run_dir.is_dir():
        raise ValueError("robustness resume requires an existing run directory")
    records_path = run_dir / "records" / "episodes.jsonl"
    if records_path.exists() or (run_dir / "robustness-outcome.json").exists() or (
        run_dir / "run_manifest.json"
    ).exists():
        raise FileExistsError("robustness resume refuses a run with existing results")

    freeze_path = run_dir / "freeze_manifest.json"
    release_path = run_dir / "release" / "test_release_manifest.json"
    selection_path = run_dir / "tuning" / "selected_lambda.json"
    training_specs_path = run_dir / "development" / "train_complete_specs.jsonl"
    test_tasks_path = run_dir / "release" / "test_tasks.jsonl"
    required_paths = (
        freeze_path,
        release_path,
        selection_path,
        training_specs_path,
        test_tasks_path,
        schema_path,
        relation_config_path,
    )
    if any(not path.is_file() for path in required_paths):
        raise ValueError("robustness resume requires frozen, released, and readable inputs")

    freeze = _load_json(freeze_path)
    release = _load_json(release_path)
    selection = _load_json(selection_path)
    if (
        freeze.get("status") != "frozen"
        or freeze.get("seed") != seed
        or freeze.get("budget") != budget
        or release.get("seed") != seed
        or selection.get("seed") != seed
        or selection.get("selection_split") != "validation"
    ):
        raise ValueError("robustness resume found inconsistent freeze, release, or selection metadata")
    selected = selection.get("selected_lambda")
    if not isinstance(selected, dict) or selected.get("lambda_cost") != freeze.get(
        "selected_lambda"
    ):
        raise ValueError("robustness resume found inconsistent selected lambda")
    selected_lambda = freeze.get("selected_lambda")
    if selected_lambda not in _ROBUSTNESS_LAMBDA_CANDIDATES:
        raise ValueError("robustness resume selected lambda is outside the frozen grid")
    if (
        freeze.get("schema_hash") != _sha256(schema_path)
        or release.get("schema_hash") != _sha256(schema_path)
        or freeze.get("relation_config_hash") != _sha256(relation_config_path)
        or release.get("relation_config_hash") != _sha256(relation_config_path)
        or freeze.get("training_specs_hash") != _sha256(training_specs_path)
    ):
        raise ValueError("robustness resume found a protected input hash mismatch")

    written = run_v3_policies(
        tasks_path=test_tasks_path,
        training_specs_path=training_specs_path,
        schema_path=schema_path,
        relation_config_path=relation_config_path,
        output_path=records_path,
        seed=seed,
        budget=budget,
        lambda_cost=float(selected_lambda),
    )
    records = _validate_records(
        records_path,
        expected_task_count=len(_load_jsonl(test_tasks_path, V3RequirementTask)),
    )
    if written != len(records):
        raise ValueError("robustness resume writer count does not match records")
    outcome_path = run_dir / "robustness-outcome.json"
    _write_json_once(
        outcome_path,
        {
            "analysis_label": _ANALYSIS_LABEL,
            "status": "succeeded",
            "seed": seed,
            "record_count": len(records),
            "records_hash": _sha256(records_path),
            "policy_names": list(V3_POLICY_NAMES),
            "execution_mode": "resumed_from_frozen_release",
        },
    )
    manifest = V3RobustnessRunManifest(
        seed=seed,
        budget=budget,
        selected_lambda=float(selected_lambda),
        record_count=len(records),
        policy_names=list(V3_POLICY_NAMES),
        artifact_hashes={
            "development_manifest": _sha256(run_dir / "development" / "development_manifest.json"),
            "selection": _sha256(selection_path),
            "freeze_manifest": _sha256(freeze_path),
            "release_manifest": _sha256(release_path),
            "records": _sha256(records_path),
            "outcome": _sha256(outcome_path),
        },
    )
    _write_json_once(run_dir / "run_manifest.json", manifest.model_dump(mode="json"))
    return manifest
