from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from analyze_v3_formal_diagnostics import _build_tables, run_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_HASH = "3D219CCA3159C1C42790C58C385CEB2E7A8170A085D762B1C0760C7A20DBCBA5"


def test_run_diagnostics_rejects_episode_hash_mismatch_without_writing_input(tmp_path: Path):
    """删除输入哈希门禁或提前写入输入时，本测试必须失败。"""
    episodes_path = tmp_path / "episodes.jsonl"
    original = b'{"not":"a formal record"}\n'
    episodes_path.write_bytes(original)

    with pytest.raises(ValueError, match="SHA-256"):
        run_diagnostics(
            episodes_path=episodes_path,
            tasks_path=tmp_path / "tasks.jsonl",
            training_specs_path=tmp_path / "training.jsonl",
            schema_path=tmp_path / "fields.yaml",
            relation_config_path=tmp_path / "relations.yaml",
            outcome_path=tmp_path / "outcome.json",
            primary_summary_path=tmp_path / "primary.json",
            output_dir=tmp_path / "diagnostics",
            expected_episodes_sha256="0" * 64,
        )

    assert episodes_path.read_bytes() == original
    assert not (tmp_path / "diagnostics").exists()


def test_run_diagnostics_accepts_the_matching_hash_before_loading_records(tmp_path: Path):
    """哈希门禁应精确接受匹配值，避免把所有调用都误判为篡改。"""
    episodes_path = tmp_path / "episodes.jsonl"
    original = b'{"not":"a formal record"}\n'
    episodes_path.write_bytes(original)

    with pytest.raises(ValueError, match="outcome"):
        run_diagnostics(
            episodes_path=episodes_path,
            tasks_path=tmp_path / "tasks.jsonl",
            training_specs_path=tmp_path / "training.jsonl",
            schema_path=tmp_path / "fields.yaml",
            relation_config_path=tmp_path / "relations.yaml",
            outcome_path=tmp_path / "outcome.json",
            primary_summary_path=tmp_path / "primary.json",
            output_dir=tmp_path / "diagnostics",
            expected_episodes_sha256=hashlib.sha256(original).hexdigest(),
        )

    assert episodes_path.read_bytes() == original


def test_run_diagnostics_exports_auditable_slices_from_frozen_formal_data(tmp_path: Path):
    """删除正式记录审计、TR-MLA 筛选或任一误差切片时，本测试必须失败。"""
    frozen_episodes_path = PROJECT_ROOT / "results" / "v3" / "formal" / "episodes.jsonl"
    episodes_path = tmp_path / "episodes.jsonl"
    shutil.copyfile(frozen_episodes_path, episodes_path)
    before = episodes_path.read_bytes()
    output_dir = tmp_path / "diagnostics"

    result = run_diagnostics(
        episodes_path=episodes_path,
        tasks_path=PROJECT_ROOT / "data" / "generated" / "v3" / "test_tasks.jsonl",
        training_specs_path=PROJECT_ROOT / "data" / "generated" / "v3" / "train_complete_specs.jsonl",
        schema_path=PROJECT_ROOT / "configs" / "fields.yaml",
        relation_config_path=PROJECT_ROOT / "configs" / "v3_relations.yaml",
        outcome_path=PROJECT_ROOT
        / "results"
        / "v3"
        / "formal"
        / "episodes.jsonl.formal-attempt.json.audit"
        / "outcome.json",
        primary_summary_path=PROJECT_ROOT
        / "results"
        / "v3"
        / "formal"
        / "summary"
        / "confirmatory_primary.json",
        output_dir=output_dir,
        expected_episodes_sha256=FORMAL_HASH,
    )

    table = pd.read_csv(output_dir / "formal_diagnostic_tables.csv")
    provenance = json.loads((output_dir / "formal_diagnostic_provenance.json").read_text(encoding="utf-8"))

    assert result["formal_record_count"] == 1800
    assert result["tr_mla_record_count"] == 180
    assert result["inferred_total"] > 0
    assert result["inferred_correct"] == result["inferred_total"]
    assert result["micro_precision"] == pytest.approx(1.0)
    assert result["episode_precision_mean"] < result["micro_precision"]
    assert result["micro_precision"] == pytest.approx(
        result["inferred_correct"] / result["inferred_total"]
    )
    assert set(table["table"]) >= {
        "inference_overall",
        "inference_by_missing_rate",
        "inference_by_mechanism",
        "inference_by_pivot",
        "inference_by_field",
    }
    assert provenance["episodes_sha256"].upper() == FORMAL_HASH
    assert provenance["formal_record_count"] == 1800
    assert episodes_path.read_bytes() == before
    assert hashlib.sha256(frozen_episodes_path.read_bytes()).hexdigest().upper() == FORMAL_HASH


def test_field_slice_counts_each_inference_once_when_an_episode_has_two_fields():
    """字段切片若回聚合整条 episode，会把第二个字段重复计入。"""
    episode_frame = pd.DataFrame(
        [
            {
                "task_id": "task-1",
                "spec_id": "spec-1",
                "missing_rate": 0.5,
                "mechanism": "interface_compatibility",
                "pivot": True,
                "episode_inferred_total": 2,
                "episode_inferred_correct": 2,
                "episode_precision": 1.0,
            },
            {
                "task_id": "task-2",
                "spec_id": "spec-2",
                "missing_rate": 0.5,
                "mechanism": "interface_compatibility",
                "pivot": False,
                "episode_inferred_total": 1,
                "episode_inferred_correct": 1,
                "episode_precision": 1.0,
            },
        ]
    )
    field_frame = pd.DataFrame(
        [
            {
                "task_id": "task-1",
                "missing_rate": 0.5,
                "mechanism": "interface_compatibility",
                "pivot": True,
                "field_id": "field_a",
                "field_kind": "categorical",
                "correct": 1,
            },
            {
                "task_id": "task-1",
                "missing_rate": 0.5,
                "mechanism": "interface_compatibility",
                "pivot": True,
                "field_id": "field_b",
                "field_kind": "categorical",
                "correct": 1,
            },
            {
                "task_id": "task-2",
                "missing_rate": 0.5,
                "mechanism": "interface_compatibility",
                "pivot": False,
                "field_id": "field_b",
                "field_kind": "categorical",
                "correct": 1,
            },
        ]
    )

    table = _build_tables(episode_frame, field_frame)
    fields = table.query("table == 'inference_by_field'").set_index("field_id")

    assert fields.loc["field_a", "inferred_total"] == 1
    assert fields.loc["field_b", "inferred_total"] == 2
    assert fields["inferred_total"].sum() == 3
