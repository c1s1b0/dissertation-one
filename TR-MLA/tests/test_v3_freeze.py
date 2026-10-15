from __future__ import annotations

import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

import paper1_active_req.experiment.v3_runner as v3_runner
from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.v3_runner import (
    V3ExperimentRecord,
    freeze_v3_artifacts,
    run_v3_selected_experiment,
)
from paper1_active_req.experiment.v3_episode import V3EpisodeResult
from paper1_active_req.generator.complete_specs import CompleteSpec, generate_complete_spec
from paper1_active_req.generator.v3_relational_specs import V3AuditMetadata, V3RequirementTask
from paper1_active_req.solver.engine import ConstraintEngine


SCHEMA_PATH = Path("configs/fields.yaml")
RELATION_CONFIG_PATH = Path("configs/v3_relations.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_SOURCES = v3_runner.formal_v3_source_paths()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze_inputs(tmp_path: Path, *, selection_split: str = "validation") -> dict[str, Path]:
    schema = load_field_schema(SCHEMA_PATH)
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(31)
    training = [
        generate_complete_spec(rng, schema, engine, f"train-{index}").model_copy(
            update={"split": "train"}
        )
        for index in range(3)
    ]
    training = [
        training[index % len(training)].model_copy(update={"spec_id": f"train-{index:03d}"})
        for index in range(180)
    ]
    training_path = tmp_path / "train_complete_specs.jsonl"
    training_path.write_text(
        "\n".join(spec.model_dump_json() for spec in training) + "\n",
        encoding="utf-8",
    )
    development_manifest = tmp_path / "development_manifest.json"
    development_manifest.write_text(
        json.dumps(
            {
                "seed": 20260806,
                "schema_hash": _sha256(SCHEMA_PATH),
                "relation_config_hash": _sha256(RELATION_CONFIG_PATH),
            }
        ),
        encoding="utf-8",
    )
    values = {field_id: value.model_copy(deep=True) for field_id, value in training[0].values.items()}
    values["workpiece_mass_kg"] = FieldValue(value=None, status=FieldStatus.HIDDEN, source="validation")
    validation_tasks = [
        V3RequirementTask(
            task_id=f"validation-{index:02d}-m{int(rate * 100)}",
            spec_id=f"validation-{index:02d}", split="validation", missing_rate=rate,
            initial_state=RequirementState(values={key: value.model_copy(deep=True) for key, value in values.items()}),
            ground_truth={key: value.model_copy(deep=True) for key, value in training[0].values.items()},
            audit=V3AuditMetadata(module_id="v3.interface_compatibility", pivot_field_id=None, mechanism="interface_compatibility"),
        )
        for index in range(60) for rate in (0.3, 0.5, 0.7)
    ]
    validation_path = tmp_path / "validation_tasks.jsonl"
    validation_path.write_text("\n".join(task.model_dump_json() for task in validation_tasks) + "\n", encoding="utf-8")
    development_manifest.write_text(
        json.dumps({
            "complete_spec_count": 240, "task_count": 720,
            "split_spec_counts": {"train": 180, "validation": 60},
            "train_spec_ids": [spec.spec_id for spec in training],
            "validation_spec_ids": [f"validation-{index:02d}" for index in range(60)],
            "test_spec_ids": [], "seed": 20260806,
            "schema_hash": _sha256(SCHEMA_PATH), "relation_config_hash": _sha256(RELATION_CONFIG_PATH),
        }), encoding="utf-8",
    )
    selection = tmp_path / "selected_lambda.json"
    experiment_config = tmp_path / "experiment.yaml"
    experiment_config.write_text(
        "master_seed: 20260806\nbudget: 8\nlambda_candidates: [0.0, 0.05, 0.1, 0.2]\n",
        encoding="utf-8",
    )
    model_hash = v3_runner._json_hash([spec.model_dump(mode="json") for spec in training])
    candidates = []
    for lambda_cost in (0.0, 0.05, 0.1, 0.2):
        config_hash = v3_runner._json_hash({
            "schema_hash": _sha256(SCHEMA_PATH), "relation_config_hash": _sha256(RELATION_CONFIG_PATH),
            "seed": 20260806, "budget": 8, "lambda_cost": lambda_cost,
            "policy_names": ["tr_mla"],
        })
        candidates.append({
            "lambda_cost": lambda_cost, "policy_names": ["tr_mla"],
            "records": [
                {"task_id": task.task_id, "spec_id": task.spec_id, "split": "validation",
                 "missing_rate": task.missing_rate, "policy_name": "tr_mla",
                 "config_hash": config_hash, "model_hash": model_hash,
                 "metrics": {"crr_at_8": 0.5, "crr_auc": 0.5, "answer_cost": 1.0}}
                for task in validation_tasks
            ],
        })
    selection.write_text(json.dumps({
        "selection_split": selection_split, "seed": 20260806, "budget": 8,
        "selected_lambda": {"lambda_cost": 0.0}, "selection_metrics": {"crr_at_8": 0.5, "crr_auc": 0.5, "answer_cost": 1.0}, "model_summary_hash": v3_runner._json_hash(sorted([model_hash] * 720)),
        "validation_tasks_hash": _sha256(validation_path), "training_specs_hash": _sha256(training_path),
        "schema_hash": _sha256(SCHEMA_PATH), "relation_config_hash": _sha256(RELATION_CONFIG_PATH),
        "experiment_config_hash": _sha256(experiment_config), "candidate_records_hash": v3_runner._json_hash(candidates),
        "candidates": candidates,
    }), encoding="utf-8")
    source = tmp_path / "runner_source.py"
    source.write_text("SOURCE = 'v3'\n", encoding="utf-8")
    return {
        "development_manifest": development_manifest,
        "selection": selection,
        "experiment_config": experiment_config,
        "source": source,
        "training": training_path,
    }


def test_freeze_manifest_requires_validation_selection_and_hashes(tmp_path: Path):
    """若接受非 validation 选择，冻结会把 test 或其他 split 的信息带入模型选择。"""
    paths = _freeze_inputs(tmp_path, selection_split="test")

    with pytest.raises(ValueError, match="validation"):
        freeze_v3_artifacts(
            development_manifest_path=paths["development_manifest"],
            selection_file=paths["selection"],
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            experiment_config_path=paths["experiment_config"],
            source_paths=FORMAL_SOURCES,
            output_path=tmp_path / "freeze.json",
        )


def test_freeze_rejects_forged_selection_seed_empty_candidates_and_provenance(tmp_path: Path):
    """freeze 必须拒绝错误 seed、空候选和任一来源哈希伪造。"""
    paths = _freeze_inputs(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selection["seed"] = 1
    paths["selection"].write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="seed/budget"):
        freeze_v3_artifacts(
            development_manifest_path=paths["development_manifest"], selection_file=paths["selection"],
            schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
            experiment_config_path=paths["experiment_config"], source_paths=FORMAL_SOURCES,
            output_path=tmp_path / "freeze.json",
        )


def test_freeze_recomputes_tie_break_selected_lambda_from_candidate_records(tmp_path: Path):
    """四组相同候选按冻结平局规则必须选择 0.0，手写 0.05 不能冻结。"""
    paths = _freeze_inputs(tmp_path)
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selection["selected_lambda"] = {"lambda_cost": 0.05}
    paths["selection"].write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="lambda/summary"):
        freeze_v3_artifacts(
            development_manifest_path=paths["development_manifest"], selection_file=paths["selection"],
            schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
            experiment_config_path=paths["experiment_config"], source_paths=FORMAL_SOURCES,
            output_path=tmp_path / "freeze.json",
        )


def test_freeze_manifest_records_all_required_hashes_and_status(tmp_path: Path):
    """若删去任一工件哈希或冻结状态，release/formal 无法复核同一实验配置。"""
    paths = _freeze_inputs(tmp_path)
    manifest = freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=tmp_path / "freeze.json",
    )

    assert manifest.status == "frozen"
    assert manifest.test_released is False
    assert manifest.selected_lambda == 0.0
    assert all(
        getattr(manifest, field)
        for field in (
            "development_manifest_hash",
            "schema_hash",
            "relation_config_hash",
            "experiment_config_hash",
            "source_hash",
            "model_summary_hash",
            "selection_file_hash",
        )
    )


def test_formal_runner_rejects_any_frozen_artifact_hash_mismatch(tmp_path: Path):
    """若不重算冻结工件哈希，冻结后改动源码仍可产生形式化结果。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    paths["selection"].write_text("{}", encoding="utf-8")
    (tmp_path / "test_tasks.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "training_specs.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze"):
        run_v3_selected_experiment(
            test_tasks_path=tmp_path / "test_tasks.jsonl",
            training_specs_path=tmp_path / "training_specs.jsonl",
            freeze_manifest_path=freeze_path,
            output_path=tmp_path / "formal" / "episodes.jsonl",
        )


def test_freeze_binds_the_exact_training_file_and_deterministic_model_summary(tmp_path: Path):
    """若 freeze 未绑定训练 JSONL，冻结后替换仍为 train 的数据会悄然改变正式模型。"""
    paths = _freeze_inputs(tmp_path)
    manifest = freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=tmp_path / "freeze.json",
    )

    assert manifest.training_specs_hash == _sha256(paths["training"])
    assert manifest.model_summary_hash != "a" * 64


def test_freeze_rejects_a_cli_selected_incomplete_source_roster(tmp_path: Path):
    """若调用方可缩减 source_paths，未列出的 baselines/lookahead 变更不会进入 freeze。"""
    paths = _freeze_inputs(tmp_path)

    with pytest.raises(ValueError, match="formal source"):
        freeze_v3_artifacts(
            development_manifest_path=paths["development_manifest"],
            selection_file=paths["selection"],
            schema_path=SCHEMA_PATH,
            relation_config_path=RELATION_CONFIG_PATH,
            experiment_config_path=paths["experiment_config"],
            source_paths=(paths["source"],),
            output_path=tmp_path / "freeze.json",
        )


def test_formal_runner_rejects_replaced_training_file_before_running(tmp_path: Path):
    """若 formal 只检查 split=train，freeze 后换一份训练集仍会进入十策略运行。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    paths["training"].write_text(
        paths["training"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    (tmp_path / "test_tasks.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="freeze.*training"):
        run_v3_selected_experiment(
            test_tasks_path=tmp_path / "test_tasks.jsonl",
            training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path,
            output_path=tmp_path / "formal" / "episodes.jsonl",
        )


def test_formal_runner_rejects_test_task_file_not_hashed_by_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """若 release manifest 不绑定 tasks/specs 哈希，伪造的 split=test 任务可冒充已发布数据。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    spec = CompleteSpec.model_validate_json(
        paths["training"].read_text(encoding="utf-8").splitlines()[0]
    ).model_copy(update={"spec_id": "forged-test", "split": "test"})
    values = {field_id: value.model_copy(deep=True) for field_id, value in spec.values.items()}
    values["workpiece_mass_kg"] = FieldValue(value=None, status=FieldStatus.HIDDEN, source="test")
    task = V3RequirementTask(
        task_id="forged-test__m50",
        spec_id=spec.spec_id,
        split="test",
        missing_rate=0.5,
        initial_state=RequirementState(values=values),
        ground_truth={field_id: value.model_copy(deep=True) for field_id, value in spec.values.items()},
        audit=V3AuditMetadata(module_id="v3.interface_compatibility", pivot_field_id=None, mechanism="interface_compatibility"),
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    specs_path = tmp_path / "test_complete_specs.jsonl"
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    specs_path.write_text(spec.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "test_release_manifest.json").write_text(
        json.dumps({"freeze_manifest_hash": _sha256(freeze_path)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        v3_runner,
        "_run_v3_policies",
        lambda **_: pytest.fail("unhashed test task reached formal execution"),
    )

    with pytest.raises(ValueError, match="release"):
        run_v3_selected_experiment(
            test_tasks_path=tasks_path,
            training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path,
            output_path=tmp_path / "formal" / "episodes.jsonl",
        )


def test_formal_runner_rejects_a_forged_release_roster_before_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """若不强制发布协议的 60 specs/180 tasks，攻击者可自造一份自洽但缩小的 release 清单。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze = freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    spec = CompleteSpec.model_validate_json(
        paths["training"].read_text(encoding="utf-8").splitlines()[0]
    ).model_copy(update={"spec_id": "forged-test", "split": "test"})
    values = {field_id: value.model_copy(deep=True) for field_id, value in spec.values.items()}
    values["workpiece_mass_kg"] = FieldValue(value=None, status=FieldStatus.HIDDEN, source="test")
    task = V3RequirementTask(
        task_id="forged-test__m50",
        spec_id=spec.spec_id,
        split="test",
        missing_rate=0.5,
        initial_state=RequirementState(values=values),
        ground_truth={field_id: value.model_copy(deep=True) for field_id, value in spec.values.items()},
        audit=V3AuditMetadata(module_id="v3.interface_compatibility", pivot_field_id=None, mechanism="interface_compatibility"),
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    specs_path = tmp_path / "test_complete_specs.jsonl"
    tasks_path.write_text(task.model_dump_json() + "\n", encoding="utf-8")
    specs_path.write_text(spec.model_dump_json() + "\n", encoding="utf-8")
    (tmp_path / "test_release_manifest.json").write_text(
        json.dumps(
            {
                "freeze_manifest_hash": _sha256(freeze_path),
                "test_tasks_hash": _sha256(tasks_path),
                "test_complete_specs_hash": _sha256(specs_path),
                "schema_hash": freeze.schema_hash,
                "relation_config_hash": freeze.relation_config_hash,
                "released_task_ids": [task.task_id],
                "released_spec_ids": [spec.spec_id],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        v3_runner,
        "_run_v3_policies",
        lambda **_: pytest.fail("forged release roster reached formal execution"),
    )

    with pytest.raises(ValueError, match="release roster"):
        run_v3_selected_experiment(
            test_tasks_path=tasks_path,
            training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path,
            output_path=tmp_path / "formal" / "episodes.jsonl",
        )


def test_formal_runner_rejects_post_freeze_baselines_source_tampering(tmp_path: Path):
    """若遗漏 baselines.py，冻结后改动该十策略依赖会逃过 formal 门禁。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"],
        selection_file=paths["selection"],
        schema_path=SCHEMA_PATH,
        relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"],
        source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    baseline_path = next(path for path in FORMAL_SOURCES if path.name == "baselines.py")
    original = baseline_path.read_bytes()
    try:
        baseline_path.write_bytes(original + b"\n")
        (tmp_path / "test_tasks.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="freeze source hash"):
            run_v3_selected_experiment(
                test_tasks_path=tmp_path / "test_tasks.jsonl",
                training_specs_path=paths["training"],
                freeze_manifest_path=freeze_path,
                output_path=tmp_path / "formal" / "episodes.jsonl",
            )
    finally:
        baseline_path.write_bytes(original)
    assert baseline_path.read_bytes() == original


def test_formal_attempt_records_failed_outcome_and_prevents_unattested_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """attempt 起点后的异常必须保留 started、failed 审计，并原样重新抛出。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"], selection_file=paths["selection"],
        schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"], source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")
    release_path = tmp_path / "test_release_manifest.json"
    release_path.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "formal" / "episodes.jsonl"

    original_load_tasks = v3_runner._load_tasks
    monkeypatch.setattr(
        v3_runner,
        "_load_tasks",
        lambda path: (_ for _ in ()).throw(RuntimeError("policy exploded"))
        if path == tasks_path else original_load_tasks(path),
    )
    with pytest.raises(RuntimeError, match="policy exploded"):
        run_v3_selected_experiment(
            test_tasks_path=tasks_path, training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path, output_path=output_path,
        )

    started = json.loads((tmp_path / "formal" / "episodes.jsonl.formal-attempt.json").read_text(encoding="utf-8"))
    failed = json.loads((tmp_path / "formal" / "episodes.jsonl.formal-attempt.json.audit" / "outcome.json").read_text(encoding="utf-8"))
    assert started["status"] == "started"
    assert failed["status"] == "failed"
    assert failed["freeze_manifest_hash"] == started["freeze_manifest_hash"]
    assert failed["release_manifest_hash"] == started["release_manifest_hash"]
    assert failed["error_summary"] == "RuntimeError: policy exploded"
    assert "timestamp" in failed
    with pytest.raises(FileExistsError, match="overwrite"):
        run_v3_selected_experiment(
            test_tasks_path=tasks_path, training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path, output_path=output_path,
        )


def test_formal_attempt_records_aborted_outcome_without_swallowing_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """中止不是成功或普通失败，且 KeyboardInterrupt 必须继续向调用者传播。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"], selection_file=paths["selection"],
        schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"], source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")
    (tmp_path / "test_release_manifest.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "formal" / "episodes.jsonl"
    original_load_tasks = v3_runner._load_tasks
    monkeypatch.setattr(
        v3_runner, "_load_tasks",
        lambda path: (_ for _ in ()).throw(KeyboardInterrupt()) if path == tasks_path else original_load_tasks(path),
    )

    with pytest.raises(KeyboardInterrupt):
        run_v3_selected_experiment(
            test_tasks_path=tasks_path, training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path, output_path=output_path,
        )

    aborted = json.loads((tmp_path / "formal" / "episodes.jsonl.formal-attempt.json.audit" / "outcome.json").read_text(encoding="utf-8"))
    assert aborted["status"] == "aborted"
    assert aborted["error_summary"] == "KeyboardInterrupt: "


def test_formal_attempt_owner_blocks_competitor_outcome_and_keeps_partial_checkpoints(tmp_path: Path):
    """错误 owner 不能提交终态；每条完成轨迹须留下不可覆盖 checkpoint。"""
    freeze = tmp_path / "freeze.json"
    release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8")
    release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / "formal" / "episodes.jsonl",
        freeze_manifest_path=freeze,
        release_manifest_path=release,
    )
    with pytest.raises(PermissionError, match="owner"):
        v3_runner._write_formal_attempt_outcome(
            attempt, owner_token="competitor", status="failed", error=RuntimeError("other")
        )
    record = V3ExperimentRecord(
        task_id="t1", spec_id="s1", split="test", missing_rate=0.5,
        mechanism="interface_compatibility", pivot=False, policy_name="tr_mla",
        config_hash="config", model_hash="model",
        episode=V3EpisodeResult(
            task_id="t1", policy_name="tr_mla",
            initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
            steps=[], solver_calls=0,
        ),
        metrics={},
    )
    v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=0)
    record_hash = _sha256(attempt.audit_dir / "records" / "00000.json")
    with pytest.raises(FileExistsError, match="overwrite"):
        v3_runner._write_formal_checkpoint(attempt, owner_token=attempt.owner_token, record_index=0, task_id="t1", policy_name="tr_mla", record_hash=record_hash)
    v3_runner._write_formal_attempt_outcome(
        attempt, owner_token=attempt.owner_token, status="failed", error=RuntimeError("boom")
    )
    outcome = json.loads((attempt.audit_dir / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["completed_record_count"] == 1
    assert outcome["error_summary"] == "RuntimeError: boom"


def test_formal_attempt_os_lease_and_fake_handle_reject_real_subprocess(tmp_path: Path):
    """子进程即使注入本地 owner 注册表，也不能伪造 OS lease 写审计。"""
    freeze = tmp_path / "freeze.json"
    release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8")
    release.write_text("{}", encoding="utf-8")
    checkpoint_attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / "checkpoint-attack" / "episodes.jsonl",
        freeze_manifest_path=freeze,
        release_manifest_path=release,
    )
    outcome_attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / "outcome-attack" / "episodes.jsonl",
        freeze_manifest_path=freeze,
        release_manifest_path=release,
    )
    child = r'''
import json, os, sys, msvcrt
from pathlib import Path
from paper1_active_req.experiment import v3_runner as r

def attack(started_text, lease_text, audit_text, operation):
    started=Path(started_text); lease=Path(lease_text); audit=Path(audit_text)
    fd=os.open(lease, os.O_RDWR)
    try:
        try:
            os.lseek(fd,0,0); msvcrt.locking(fd, msvcrt.LK_NBLCK, 2); locked='locked'
        except OSError:
            locked='blocked'
        fake=r.FormalAttempt(started,audit,'forged',lease,fd)
        r._FORMAL_ATTEMPT_OWNERS[started.resolve()] = fake
        try:
            if operation == 'checkpoint':
                r._write_formal_checkpoint(
                    fake, owner_token='forged', record_index=0,
                    task_id='x', policy_name='tr', record_hash='x',
                )
            else:
                r._write_formal_attempt_outcome(
                    fake, owner_token='forged', status='failed', error=RuntimeError('forged'),
                )
            write='wrote'
        except Exception as error:
            write=type(error).__name__
        return locked + ':' + write
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

print(json.dumps({
    'checkpoint': attack(*sys.argv[1:4], 'checkpoint'),
    'outcome': attack(*sys.argv[4:7], 'outcome'),
}, sort_keys=True))
'''
    attempts = (checkpoint_attempt, outcome_attempt)
    try:
        result = subprocess.run(
            [
                sys.executable, "-c", child,
                str(checkpoint_attempt.started_path), str(checkpoint_attempt.lease_path), str(checkpoint_attempt.audit_dir),
                str(outcome_attempt.started_path), str(outcome_attempt.lease_path), str(outcome_attempt.audit_dir),
            ],
            text=True, capture_output=True, check=True,
        )
        assert json.loads(result.stdout) == {
            "checkpoint": "blocked:PermissionError",
            "outcome": "blocked:PermissionError",
        }
        assert not (checkpoint_attempt.audit_dir / "checkpoints" / "00000.json").exists()
        assert not (outcome_attempt.audit_dir / "outcome.json").exists()
    finally:
        # 兼容红灯阶段的单个 2-byte range lock 与绿灯后的两个 1-byte lock。
        for attempt in attempts:
            for offset in (0, 1):
                try:
                    os.lseek(attempt.lease_fd, offset, os.SEEK_SET)
                    msvcrt.locking(attempt.lease_fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            try:
                os.lseek(attempt.lease_fd, 0, os.SEEK_SET)
                msvcrt.locking(attempt.lease_fd, msvcrt.LK_UNLCK, 2)
            except OSError:
                pass
            try:
                os.close(attempt.lease_fd)
            except OSError:
                pass
            v3_runner._FORMAL_ATTEMPT_OWNERS.pop(attempt.started_path.resolve(), None)


def test_formal_attempt_begin_failures_release_full_os_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """取得 lease 后任一建档失败都必须释放全部字节并关闭句柄。"""
    freeze = tmp_path / "freeze.json"
    release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8")
    release.write_text("{}", encoding="utf-8")
    original_write_json_once = v3_runner._write_json_once
    probe = r'''
import msvcrt, os, sys
fd=os.open(sys.argv[1], os.O_RDWR)
try:
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_NBLCK, 2)
    print('acquired')
    os.lseek(fd, 0, os.SEEK_SET)
    msvcrt.locking(fd, msvcrt.LK_UNLCK, 2)
finally:
    os.close(fd)
'''

    def assert_released(output_path: Path) -> None:
        started = output_path.with_name(f"{output_path.name}.formal-attempt.json")
        lease = started.with_name(f"{started.name}.lease")
        result = subprocess.run(
            [sys.executable, "-c", probe, str(lease)],
            text=True, capture_output=True, check=True,
        )
        assert result.stdout.strip() == "acquired"
        lease.unlink()  # Windows 上若 begin 遗留打开句柄，此处会以 WinError 32 失败。
        assert not lease.exists()

    started_output = tmp_path / "started-exists" / "episodes.jsonl"
    started = started_output.with_name(f"{started_output.name}.formal-attempt.json")
    started.parent.mkdir(parents=True)
    started.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        v3_runner._begin_formal_attempt(
            output_path=started_output,
            freeze_manifest_path=freeze,
            release_manifest_path=release,
        )
    assert_released(started_output)

    directory_output = tmp_path / "audit-dir-exists" / "episodes.jsonl"
    directory_started = directory_output.with_name(f"{directory_output.name}.formal-attempt.json")
    directory_started.with_name(f"{directory_started.name}.audit").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        v3_runner._begin_formal_attempt(
            output_path=directory_output,
            freeze_manifest_path=freeze,
            release_manifest_path=release,
        )
    assert_released(directory_output)

    writer_output = tmp_path / "started-write-fails" / "episodes.jsonl"
    monkeypatch.setattr(
        v3_runner,
        "_write_json_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("started write failed")),
    )
    with pytest.raises(RuntimeError, match="started write failed"):
        v3_runner._begin_formal_attempt(
            output_path=writer_output,
            freeze_manifest_path=freeze,
            release_manifest_path=release,
        )
    assert_released(writer_output)

    monkeypatch.setattr(v3_runner, "_write_json_once", original_write_json_once)

    class RejectingRegistry(dict[Path, v3_runner.FormalAttempt]):
        captured_attempt: v3_runner.FormalAttempt | None = None

        def __setitem__(self, key: Path, value: v3_runner.FormalAttempt) -> None:
            self.captured_attempt = value
            raise RuntimeError("owner registry failed")

    rejecting_registry = RejectingRegistry()
    monkeypatch.setattr(v3_runner, "_FORMAL_ATTEMPT_OWNERS", rejecting_registry)
    registry_output = tmp_path / "owner-registry-fails" / "episodes.jsonl"
    try:
        with pytest.raises(RuntimeError, match="owner registry failed"):
            v3_runner._begin_formal_attempt(
                output_path=registry_output,
                freeze_manifest_path=freeze,
                release_manifest_path=release,
            )
        assert_released(registry_output)
    finally:
        # 红灯实现把注册放在释放保护块之外；捕获句柄只用于保证测试可清理。
        attempt = rejecting_registry.captured_attempt
        if attempt is not None:
            for offset in (0, 1):
                try:
                    os.lseek(attempt.lease_fd, offset, os.SEEK_SET)
                    msvcrt.locking(attempt.lease_fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            try:
                os.close(attempt.lease_fd)
            except OSError:
                pass


@pytest.mark.parametrize("status,error", (
    ("failed", RuntimeError("checkpoint write failed")),
    ("aborted", KeyboardInterrupt("checkpoint write aborted")),
))
def test_non_success_outcome_preserves_record_written_before_checkpoint_failure(
    tmp_path: Path, status: str, error: BaseException
):
    """record 已审计落盘但 checkpoint 未形成时，失败/中止均保留原异常和 pending 数。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(output_path=tmp_path / status / "episodes.jsonl", freeze_manifest_path=freeze, release_manifest_path=release)
    record = attempt.audit_dir / "records" / "00000.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{}", encoding="utf-8")
    v3_runner._write_formal_attempt_outcome(attempt, owner_token=attempt.owner_token, status=status, error=error)
    outcome = json.loads((attempt.audit_dir / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["completed_record_count"] == 0
    assert outcome["audit_record_count"] == 1
    assert outcome["pending_audit_record_count"] == 1
    assert outcome["error_summary"] == f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("status,error", (
    ("failed", RuntimeError("orphan checkpoint")),
    ("aborted", KeyboardInterrupt("orphan checkpoint")),
))
def test_non_success_terminal_rejects_checkpoint_without_record(
    tmp_path: Path, status: str, error: BaseException
):
    """checkpoint-only 不是 completed，也不能提交 failed/aborted terminal。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / status / "episodes.jsonl",
        freeze_manifest_path=freeze, release_manifest_path=release,
    )
    v3_runner._write_formal_checkpoint(
        attempt, owner_token=attempt.owner_token, record_index=0,
        task_id="task-1", policy_name="tr_mla", record_hash="missing",
    )

    with pytest.raises(ValueError, match="checkpoint without record"):
        v3_runner._write_formal_attempt_outcome(
            attempt, owner_token=attempt.owner_token, status=status, error=error,
        )
    assert not (attempt.audit_dir / "outcome.json").exists()


def test_success_outcome_rejects_empty_audit_without_writing_terminal(tmp_path: Path):
    """零审计或缺少最终 output 时不得伪造 succeeded outcome。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(output_path=tmp_path / "formal" / "episodes.jsonl", freeze_manifest_path=freeze, release_manifest_path=release)
    with pytest.raises(ValueError, match="1800"):
        v3_runner._write_formal_attempt_outcome(attempt, owner_token=attempt.owner_token, status="succeeded", record_count=1800, output_path=tmp_path / "formal" / "episodes.jsonl")
    assert not (attempt.audit_dir / "outcome.json").exists()


def test_terminal_state_lock_rejects_owner_checkpoint_and_record_after_outcome(tmp_path: Path):
    """终态提交与后续 checkpoint/record 的竞争由同一 state lock 串行化。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(output_path=tmp_path / "formal" / "episodes.jsonl", freeze_manifest_path=freeze, release_manifest_path=release)
    ready = threading.Event(); release_lock = threading.Event(); errors: list[BaseException] = []
    def checkpoint_writer() -> None:
        ready.set(); release_lock.wait()
        try:
            v3_runner._write_formal_checkpoint(attempt, owner_token=attempt.owner_token, record_index=0, task_id="x", policy_name="tr", record_hash="x")
        except BaseException as error:
            errors.append(error)
    with attempt.state_lock:
        worker = threading.Thread(target=checkpoint_writer); worker.start(); ready.wait()
        v3_runner._write_formal_attempt_outcome(attempt, owner_token=attempt.owner_token, status="failed", error=RuntimeError("stop"), _locked=True)
        release_lock.set()
    worker.join()
    assert errors and isinstance(errors[0], PermissionError | RuntimeError)
    assert not (attempt.audit_dir / "checkpoints" / "00000.json").exists()


def test_success_outcome_requires_matching_final_output_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """success 必须逐条比较最终 JSONL 与 audit record，不能只看数量。"""
    monkeypatch.setattr(v3_runner, "_FORMAL_TASK_COUNT", 1)
    monkeypatch.setattr(v3_runner, "_FORMAL_RECORD_COUNT", 1)
    monkeypatch.setattr(v3_runner, "V3_POLICY_NAMES", ("tr_mla",))
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    def attempt_at(name: str):
        return v3_runner._begin_formal_attempt(output_path=tmp_path / name / "formal" / "episodes.jsonl", freeze_manifest_path=freeze, release_manifest_path=release)
    record = V3ExperimentRecord(task_id="t", spec_id="s", split="test", missing_rate=0.5, mechanism="interface_compatibility", pivot=False, policy_name="tr_mla", config_hash="a", model_hash="b", episode=V3EpisodeResult(task_id="t", policy_name="tr_mla", initial_state=RequirementState(values={}), final_state=RequirementState(values={}), steps=[], solver_calls=0), metrics={})
    missing = attempt_at("missing")
    v3_runner._write_formal_record_checkpoint(missing, record=record, record_index=0)
    with pytest.raises(ValueError, match="unreadable"):
        v3_runner._write_formal_attempt_outcome(missing, owner_token=missing.owner_token, status="succeeded", record_count=1, output_path=tmp_path / "missing" / "formal" / "episodes.jsonl")
    assert not (missing.audit_dir / "outcome.json").exists()
    mismatch = attempt_at("mismatch")
    v3_runner._write_formal_record_checkpoint(mismatch, record=record, record_index=0)
    output = tmp_path / "mismatch" / "formal" / "episodes.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record.model_copy(update={"task_id": "other"}).model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="match"):
        v3_runner._write_formal_attempt_outcome(mismatch, owner_token=mismatch.owner_token, status="succeeded", record_count=1, output_path=output)
    assert not (mismatch.audit_dir / "outcome.json").exists()
    matching = attempt_at("matching")
    v3_runner._write_formal_record_checkpoint(matching, record=record, record_index=0)
    output = tmp_path / "matching" / "formal" / "episodes.jsonl"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    v3_runner._write_formal_attempt_outcome(matching, owner_token=matching.owner_token, status="succeeded", record_count=1, output_path=output)
    assert json.loads((matching.audit_dir / "outcome.json").read_text(encoding="utf-8"))["status"] == "succeeded"


@pytest.mark.parametrize("field", ("record_index", "task_id", "policy_name", "attempt_hash", "record_hash"))
def test_success_rejects_tampered_checkpoint_or_audit_envelope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str):
    """success 必须验证每个 checkpoint 和 audit envelope 的全部绑定元数据。"""
    monkeypatch.setattr(v3_runner, "_FORMAL_RECORD_COUNT", 1)
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"; freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(output_path=tmp_path / "formal" / "episodes.jsonl", freeze_manifest_path=freeze, release_manifest_path=release)
    record = V3ExperimentRecord(task_id="t", spec_id="s", split="test", missing_rate=0.5, mechanism="interface_compatibility", pivot=False, policy_name="tr_mla", config_hash="a", model_hash="b", episode=V3EpisodeResult(task_id="t", policy_name="tr_mla", initial_state=RequirementState(values={}), final_state=RequirementState(values={}), steps=[], solver_calls=0), metrics={})
    v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=0)
    cp = attempt.audit_dir / "checkpoints" / "00000.json"; data=json.loads(cp.read_text(encoding="utf-8")); data[field] = "tampered"; cp.write_text(json.dumps(data), encoding="utf-8")
    output=tmp_path / "formal" / "episodes.jsonl"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(record.model_dump_json()+"\n", encoding="utf-8")
    with pytest.raises(ValueError): v3_runner._write_formal_attempt_outcome(attempt, owner_token=attempt.owner_token, status="succeeded", record_count=1, output_path=output)
    assert not (attempt.audit_dir / "outcome.json").exists()


@pytest.mark.parametrize("status,error", (
    ("failed", RuntimeError("business-failed")),
    ("aborted", KeyboardInterrupt("business-aborted")),
))
@pytest.mark.parametrize("field", (
    "checkpoint_filename", "record_filename", "record_index", "task_id",
    "policy_name", "attempt_hash", "record_hash",
))
def test_non_success_terminal_rejects_tampered_paired_checkpoint(
    tmp_path: Path, status: str, error: BaseException, field: str
):
    """failed/aborted 也不得把任一损坏的已配对 checkpoint 计为 completed。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / status / field / "episodes.jsonl",
        freeze_manifest_path=freeze, release_manifest_path=release,
    )
    record = V3ExperimentRecord(
        task_id="task-1", spec_id="spec-1", split="test", missing_rate=0.5,
        mechanism="interface_compatibility", pivot=False, policy_name="tr_mla",
        config_hash="config", model_hash="model",
        episode=V3EpisodeResult(
            task_id="task-1", policy_name="tr_mla",
            initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
            steps=[], solver_calls=0,
        ),
        metrics={},
    )
    v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=0)
    checkpoint_path = attempt.audit_dir / "checkpoints" / "00000.json"
    if field == "checkpoint_filename":
        checkpoint_path.rename(checkpoint_path.with_name("00001.json"))
    elif field == "record_filename":
        record_path = attempt.audit_dir / "records" / "00000.json"
        record_path.rename(record_path.with_name("00001.json"))
    else:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint[field] = "tampered"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError):
        v3_runner._write_formal_attempt_outcome(
            attempt, owner_token=attempt.owner_token, status=status, error=error,
        )
    assert not (attempt.audit_dir / "outcome.json").exists()


@pytest.mark.parametrize("status,error", (
    ("failed", RuntimeError("business-failed")),
    ("aborted", KeyboardInterrupt("business-aborted")),
))
@pytest.mark.parametrize("tamper", ("audit_attempt_hash", "invalid_record_model"))
def test_non_success_terminal_rejects_tampered_paired_audit_record(
    tmp_path: Path, status: str, error: BaseException, tamper: str
):
    """已配对 record 的 attempt 归属和模型结构对所有终态均为强制不变量。"""
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / status / tamper / "episodes.jsonl",
        freeze_manifest_path=freeze, release_manifest_path=release,
    )
    record = V3ExperimentRecord(
        task_id="task-1", spec_id="spec-1", split="test", missing_rate=0.5,
        mechanism="interface_compatibility", pivot=False, policy_name="tr_mla",
        config_hash="config", model_hash="model",
        episode=V3EpisodeResult(
            task_id="task-1", policy_name="tr_mla",
            initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
            steps=[], solver_calls=0,
        ),
        metrics={},
    )
    v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=0)
    record_path = attempt.audit_dir / "records" / "00000.json"
    audit = json.loads(record_path.read_text(encoding="utf-8"))
    if tamper == "audit_attempt_hash":
        audit["attempt_hash"] = "tampered"
    else:
        audit["record"] = {"task_id": "incomplete"}
    record_path.write_text(json.dumps(audit), encoding="utf-8")
    checkpoint_path = attempt.audit_dir / "checkpoints" / "00000.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["record_hash"] = _sha256(record_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError):
        v3_runner._write_formal_attempt_outcome(
            attempt, owner_token=attempt.owner_token, status=status, error=error,
        )
    assert not (attempt.audit_dir / "outcome.json").exists()


@pytest.mark.parametrize("business_error", (
    RuntimeError("policy-business-failure"),
    KeyboardInterrupt("operator-abort"),
))
def test_public_formal_runner_preserves_original_error_when_terminal_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, business_error: BaseException
):
    """生产入口的终态审计失败只能附注，绝不能替换原始业务失败或中止。"""
    paths = _freeze_inputs(tmp_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_v3_artifacts(
        development_manifest_path=paths["development_manifest"], selection_file=paths["selection"],
        schema_path=SCHEMA_PATH, relation_config_path=RELATION_CONFIG_PATH,
        experiment_config_path=paths["experiment_config"], source_paths=FORMAL_SOURCES,
        output_path=freeze_path,
    )
    tasks_path = tmp_path / "test_tasks.jsonl"
    tasks_path.write_text("", encoding="utf-8")
    (tmp_path / "test_complete_specs.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "test_release_manifest.json").write_text("{}", encoding="utf-8")
    output_path = tmp_path / "formal" / "episodes.jsonl"

    def fail_after_tampering(*, attempt_path: v3_runner.FormalAttempt, **_kwargs: object) -> int:
        record = V3ExperimentRecord(
            task_id="task-1", spec_id="spec-1", split="test", missing_rate=0.5,
            mechanism="interface_compatibility", pivot=False, policy_name="tr_mla",
            config_hash="config", model_hash="model",
            episode=V3EpisodeResult(
                task_id="task-1", policy_name="tr_mla",
                initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
                steps=[], solver_calls=0,
            ),
            metrics={},
        )
        v3_runner._write_formal_record_checkpoint(attempt_path, record=record, record_index=0)
        checkpoint_path = attempt_path.audit_dir / "checkpoints" / "00000.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["task_id"] = "tampered"
        checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
        raise business_error

    monkeypatch.setattr(v3_runner, "_run_v3_selected_experiment_impl", fail_after_tampering)
    with pytest.raises(type(business_error)) as caught:
        run_v3_selected_experiment(
            test_tasks_path=tasks_path, training_specs_path=paths["training"],
            freeze_manifest_path=freeze_path, output_path=output_path,
        )

    assert caught.value is business_error
    assert any("formal audit outcome could not be written" in note for note in caught.value.__notes__)
    audit_dir = tmp_path / "formal" / "episodes.jsonl.formal-attempt.json.audit"
    assert not (audit_dir / "outcome.json").exists()


@pytest.mark.parametrize("invalid_roster", ("duplicate_task_policy", "missing_policy"))
def test_success_policy_coverage_gate_runs_under_reduced_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_roster: str
):
    """缩小协议仍必须触发 task×policy 唯一覆盖门，不能依赖常量恰等于 1800。"""
    monkeypatch.setattr(v3_runner, "_FORMAL_TASK_COUNT", 1)
    monkeypatch.setattr(v3_runner, "_FORMAL_RECORD_COUNT", len(v3_runner.V3_POLICY_NAMES))
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    attempt = v3_runner._begin_formal_attempt(
        output_path=tmp_path / invalid_roster / "episodes.jsonl",
        freeze_manifest_path=freeze, release_manifest_path=release,
    )
    policy_names = list(v3_runner.V3_POLICY_NAMES)
    if invalid_roster == "duplicate_task_policy":
        policy_names[-1] = policy_names[0]
    else:
        policy_names[-1] = "unexpected_policy"
    records = [
        V3ExperimentRecord(
            task_id="task-1", spec_id="spec-1", split="test", missing_rate=0.5,
            mechanism="interface_compatibility", pivot=False, policy_name=policy_name,
            config_hash="config", model_hash="model",
            episode=V3EpisodeResult(
                task_id="task-1", policy_name=policy_name,
                initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
                steps=[], solver_calls=0,
            ),
            metrics={},
        )
        for policy_name in policy_names
    ]
    for index, record in enumerate(records):
        v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=index)
    output_path = tmp_path / invalid_roster / "episodes.jsonl"
    output_path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ten policies per task"):
        v3_runner._write_formal_attempt_outcome(
            attempt, owner_token=attempt.owner_token, status="succeeded",
            record_count=len(records), output_path=output_path,
        )
    assert not (attempt.audit_dir / "outcome.json").exists()


def test_success_policy_coverage_accepts_one_task_all_ten_policies_under_reduced_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """一 task × 十个冻结 policy 各一次时，缩小协议应走同一结构门并成功。"""
    monkeypatch.setattr(v3_runner, "_FORMAL_TASK_COUNT", 1)
    monkeypatch.setattr(v3_runner, "_FORMAL_RECORD_COUNT", len(v3_runner.V3_POLICY_NAMES))
    freeze = tmp_path / "freeze.json"; release = tmp_path / "release.json"
    freeze.write_text("{}", encoding="utf-8"); release.write_text("{}", encoding="utf-8")
    output_path = tmp_path / "valid" / "episodes.jsonl"
    attempt = v3_runner._begin_formal_attempt(
        output_path=output_path, freeze_manifest_path=freeze, release_manifest_path=release,
    )
    records = [
        V3ExperimentRecord(
            task_id="task-1", spec_id="spec-1", split="test", missing_rate=0.5,
            mechanism="interface_compatibility", pivot=False, policy_name=policy_name,
            config_hash="config", model_hash="model",
            episode=V3EpisodeResult(
                task_id="task-1", policy_name=policy_name,
                initial_state=RequirementState(values={}), final_state=RequirementState(values={}),
                steps=[], solver_calls=0,
            ),
            metrics={},
        )
        for policy_name in v3_runner.V3_POLICY_NAMES
    ]
    for index, record in enumerate(records):
        v3_runner._write_formal_record_checkpoint(attempt, record=record, record_index=index)
    output_path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")

    v3_runner._write_formal_attempt_outcome(
        attempt, owner_token=attempt.owner_token, status="succeeded",
        record_count=len(records), output_path=output_path,
    )

    outcome = json.loads((attempt.audit_dir / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "succeeded"
    assert outcome["completed_record_count"] == len(v3_runner.V3_POLICY_NAMES)
