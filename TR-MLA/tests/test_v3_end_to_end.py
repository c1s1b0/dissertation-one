from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

import paper1_active_req.evaluation.v3_statistics as v3_statistics
import paper1_active_req.experiment.v3_runner as v3_runner
import paper1_active_req.generator.v3_relational_specs as v3_generator
from paper1_active_req.experiment.v3_runner import (
    formal_v3_source_paths,
    freeze_v3_artifacts,
    run_v3_selected_experiment,
    run_v3_validation_tuning,
    summarize_v3_results,
)
from paper1_active_req.generator.v3_relational_specs import (
    generate_v3_development_dataset,
    release_v3_test_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "configs" / "fields.yaml"
RELATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "v3_relations.yaml"


@dataclass(frozen=True)
class V3SmokeArtifacts:
    development: object
    validation_selection: dict[str, object]
    freeze_manifest: object
    records_written: int
    summary_dir: Path
    dataset_dir: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_file_hashes(root: Path) -> dict[Path, str]:
    return {
        path: _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _require_validation_selection(selection: dict[str, object]) -> None:
    assert selection["selection_split"] == "validation", (
        "v3 smoke requires a validation selection split"
    )


def run_v3_end_to_end_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> V3SmokeArtifacts:
    """仅替换私有协议常量的 12/6/6 临时闭环；生产 CLI 无缩小入口。"""
    with monkeypatch.context() as smoke_patch:
        smoke_patch.setattr(v3_generator, "_DEVELOPMENT_TRAIN_SPEC_COUNT", 12)
        smoke_patch.setattr(v3_generator, "_DEVELOPMENT_VALIDATION_SPEC_COUNT", 6)
        smoke_patch.setattr(v3_generator, "_TEST_SPECS_PER_MECHANISM", 2)
        smoke_patch.setattr(v3_generator, "_TEST_PIVOT_SPECS_PER_MECHANISM", 1)
        smoke_patch.setattr(v3_runner, "_PROTOCOL_MASTER_SEED", 20260806)
        smoke_patch.setattr(v3_runner, "_PROTOCOL_BUDGET", 2)
        smoke_patch.setattr(v3_runner, "_DEVELOPMENT_TRAIN_SPEC_COUNT", 12)
        smoke_patch.setattr(v3_runner, "_DEVELOPMENT_VALIDATION_SPEC_COUNT", 6)
        smoke_patch.setattr(v3_runner, "_DEVELOPMENT_SPEC_COUNT", 18)
        smoke_patch.setattr(v3_runner, "_DEVELOPMENT_TASK_COUNT", 54)
        smoke_patch.setattr(v3_runner, "_VALIDATION_TASK_COUNT", 18)
        smoke_patch.setattr(v3_runner, "_FORMAL_TEST_SPEC_COUNT", 6)
        smoke_patch.setattr(v3_runner, "_FORMAL_TASK_COUNT", 18)
        smoke_patch.setattr(v3_runner, "_FORMAL_RECORD_COUNT", 180)
        smoke_patch.setattr(v3_statistics, "_PRIMARY_SPEC_COUNT", 6)
        return _run_v3_end_to_end_smoke_with_protocol(tmp_path)


def _run_v3_end_to_end_smoke_with_protocol(tmp_path: Path) -> V3SmokeArtifacts:

    dataset_dir = tmp_path / "dataset"
    development = generate_v3_development_dataset(
        master_seed=20260806,
        output_dir=dataset_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
    )
    experiment_config = tmp_path / "v3_experiment_smoke.yaml"
    experiment_config.write_text(
        "master_seed: 20260806\nbudget: 2\nlambda_candidates: [0.0, 0.05, 0.1, 0.2]\n",
        encoding="utf-8",
    )
    tuning_dir = tmp_path / "tuning"
    selection = run_v3_validation_tuning(
        validation_tasks_path=dataset_dir / "validation_tasks.jsonl",
        training_specs_path=dataset_dir / "train_complete_specs.jsonl",
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        output_dir=tuning_dir,
        seed=20260806,
        budget=2,
        experiment_config_path=experiment_config,
    )
    _require_validation_selection(selection)
    frozen = freeze_v3_artifacts(
        development_manifest_path=dataset_dir / "development_manifest.json",
        selection_file=tuning_dir / "selected_lambda.json",
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=experiment_config,
        source_paths=formal_v3_source_paths(),
        output_path=tmp_path / "freeze.json",
    )
    release_v3_test_dataset(
        master_seed=20260806,
        output_dir=dataset_dir,
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        freeze_manifest_path=tmp_path / "freeze.json",
    )
    episodes_path = tmp_path / "formal" / "episodes.jsonl"
    written = run_v3_selected_experiment(
        test_tasks_path=dataset_dir / "test_tasks.jsonl",
        training_specs_path=dataset_dir / "train_complete_specs.jsonl",
        freeze_manifest_path=tmp_path / "freeze.json",
        output_path=episodes_path,
    )
    summary_dir = tmp_path / "summary"
    summarize_v3_results(episodes_path=episodes_path, output_dir=summary_dir)
    return V3SmokeArtifacts(
        development=development,
        validation_selection=selection,
        freeze_manifest=frozen,
        records_written=written,
        summary_dir=summary_dir,
        dataset_dir=dataset_dir,
    )


def test_v3_smoke_rejects_nonvalidation_selection_before_freeze():
    """selection 只能来自 validation，且无需替换任何 E2E 阶段函数。"""
    with pytest.raises(AssertionError, match="selection split"):
        _require_validation_selection({"selection_split": "test"})


@pytest.mark.filterwarnings(
    "ignore:Sample size too small for normal approximation.:UserWarning"
)
def test_v3_end_to_end_smoke_isolated_to_tmp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """若任一阶段跳过冻结门禁、混入生产路径或遗漏十策略，临时 smoke 将无法留下可审计闭环。"""
    production_v3_paths = (
        PROJECT_ROOT / "data" / "generated" / "v3",
        PROJECT_ROOT / "results" / "v3" / "tuning",
        PROJECT_ROOT / "results" / "v3" / "formal",
    )
    production_v3_before = {
        path: _formal_file_hashes(path)
        for path in production_v3_paths
    }
    v1_v2_before = {
        version: _formal_file_hashes(PROJECT_ROOT / "results" / version / "formal")
        for version in ("v1", "v2")
    }

    artifacts = run_v3_end_to_end_smoke(tmp_path, monkeypatch)
    assert artifacts.development.split_spec_counts == {"train": 12, "validation": 6}
    released_specs = [
        json.loads(line)
        for line in (artifacts.dataset_dir / "test_complete_specs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    released_tasks = [
        json.loads(line)
        for line in (artifacts.dataset_dir / "test_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    audit_by_spec = {
        task["spec_id"]: task["audit"]
        for task in released_tasks
    }
    assert len(released_specs) == 6
    assert {spec["spec_id"] for spec in released_specs} == set(audit_by_spec)
    assert Counter(
        audit["mechanism"] for audit in audit_by_spec.values()
    ) == {
        "interface_compatibility": 2,
        "continuous_clearance": 2,
        "conditional_tolerance": 2,
    }
    assert Counter(
        (audit["mechanism"], audit["pivot_field_id"] is not None)
        for audit in audit_by_spec.values()
    ) == {
        ("interface_compatibility", True): 1,
        ("interface_compatibility", False): 1,
        ("continuous_clearance", True): 1,
        ("continuous_clearance", False): 1,
        ("conditional_tolerance", True): 1,
        ("conditional_tolerance", False): 1,
    }
    assert artifacts.validation_selection["selection_split"] == "validation"
    assert artifacts.freeze_manifest.status == "frozen"
    assert artifacts.freeze_manifest.test_released is False
    assert artifacts.freeze_manifest.budget == 2
    assert artifacts.records_written == 6 * 3 * 10
    assert (artifacts.summary_dir / "overall.csv").exists()
    assert (artifacts.summary_dir / "primary_m50_m70.csv").exists()
    assert (
        v3_generator._DEVELOPMENT_TRAIN_SPEC_COUNT,
        v3_generator._DEVELOPMENT_VALIDATION_SPEC_COUNT,
        v3_generator._TEST_SPECS_PER_MECHANISM,
        v3_generator._TEST_PIVOT_SPECS_PER_MECHANISM,
    ) == (180, 60, 20, 15)

    assert {
        path: _formal_file_hashes(path)
        for path in production_v3_paths
    } == production_v3_before
    assert {
        version: _formal_file_hashes(PROJECT_ROOT / "results" / version / "formal")
        for version in ("v1", "v2")
    } == v1_v2_before
