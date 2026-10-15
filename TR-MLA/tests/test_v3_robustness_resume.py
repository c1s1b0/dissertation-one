from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper1_active_req.experiment.v3_robustness import resume_v3_robustness_run


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sidecar_root(tmp_path: Path) -> Path:
    return tmp_path / "research_artifacts" / "v3_robustness_20260810"


def test_resume_requires_frozen_released_run_without_records(tmp_path: Path) -> None:
    root = _sidecar_root(tmp_path)
    run_dir = root / "runs" / "seed-20260813"
    _write_json(
        run_dir / "freeze_manifest.json",
        {
            "status": "frozen",
            "seed": 20260813,
            "budget": 8,
            "selected_lambda": 0.0,
            "schema_hash": "schema",
            "relation_config_hash": "relations",
            "training_specs_hash": "training",
        },
    )
    _write_json(
        run_dir / "release" / "test_release_manifest.json",
        {
            "seed": 20260813,
            "schema_hash": "schema",
            "relation_config_hash": "relations",
        },
    )
    _write_json(
        run_dir / "tuning" / "selected_lambda.json",
        {"seed": 20260813, "selection_split": "validation", "selected_lambda": {"lambda_cost": 0.0}},
    )
    (run_dir / "release" / "test_tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "development").mkdir()
    (run_dir / "development" / "train_complete_specs.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="readable inputs"):
        resume_v3_robustness_run(
            root=root,
            seed=20260813,
            schema_path=tmp_path / "schema.yaml",
            relation_config_path=tmp_path / "relations.yaml",
        )


def test_resume_rejects_existing_records_before_any_execution(tmp_path: Path) -> None:
    root = _sidecar_root(tmp_path)
    run_dir = root / "runs" / "seed-20260813"
    (run_dir / "records").mkdir(parents=True)
    (run_dir / "records" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="existing results"):
        resume_v3_robustness_run(
            root=root,
            seed=20260813,
            schema_path=tmp_path / "schema.yaml",
            relation_config_path=tmp_path / "relations.yaml",
        )
