from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

import paper1_active_req.experiment.v3_runner as v3_runner


def test_robustness_protocol_module_exposes_predeclared_seeds() -> None:
    """删除 sidecar 协议模块会使独立种子来源不再受预声明约束。"""
    specification = importlib.util.find_spec(
        "paper1_active_req.experiment.v3_robustness"
    )

    assert specification is not None, "v3 robustness protocol module is required"
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    assert module.ROBUSTNESS_SEEDS == (20260813, 20260817, 20260823)


def test_prepare_run_requires_fixed_seed_and_isolated_sidecar_root(
    tmp_path: Path,
) -> None:
    """删除根目录或不可覆盖保护会使鲁棒性记录污染正式结果或被重用。"""
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    root = tmp_path / "research_artifacts" / "v3_robustness_20260810"

    run_dir = module.prepare_v3_robustness_run(root=root, seed=20260813)

    assert run_dir == root / "runs" / "seed-20260813"
    assert run_dir.is_dir()
    with pytest.raises(ValueError, match="seed"):
        module.prepare_v3_robustness_run(root=root, seed=20260814)
    with pytest.raises(ValueError, match="sidecar"):
        module.prepare_v3_robustness_run(root=tmp_path / "wrong-root", seed=20260817)
    with pytest.raises(FileExistsError):
        module.prepare_v3_robustness_run(root=root, seed=20260813)


def test_robustness_protocol_is_immutable_and_declares_exploratory_scope(
    tmp_path: Path,
) -> None:
    """协议若不可独立落盘，后续三种子结果便不能证明其预先声明。"""
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    root = tmp_path / "research_artifacts" / "v3_robustness_20260810"

    protocol = module.write_v3_robustness_protocol(root=root)

    assert protocol["analysis_label"] == "independent_robustness_exploratory"
    assert protocol["formal_pooling"] == "prohibited"
    assert protocol["seeds"] == [20260813, 20260817, 20260823]
    assert protocol["test_specification"] == {
        "total": 60,
        "per_mechanism": 20,
        "pivot_per_mechanism": 15,
        "control_per_mechanism": 5,
    }
    with pytest.raises(FileExistsError):
        module.write_v3_robustness_protocol(root=root)


def test_cross_seed_summary_lists_effects_without_a_pooled_p_value(tmp_path: Path) -> None:
    """跨种子汇总只能并排陈列各自效应，不能伪造新的合并显著性检验。"""
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    root = tmp_path / "research_artifacts" / "v3_robustness_20260810"
    module.write_v3_robustness_protocol(root=root)
    for seed in module.ROBUSTNESS_SEEDS:
        effect_path = root / "runs" / f"seed-{seed}" / "summary" / "robustness_effect.json"
        effect_path.parent.mkdir(parents=True)
        effect_path.write_text(
            json.dumps(
                {
                    "analysis_label": "independent_robustness_exploratory",
                    "seed": seed,
                    "primary": {"n": 60, "mean_difference": 0.1, "ci_low": 0.01, "ci_high": 0.2},
                    "high_pressure_m70_pivot": {"n": 45, "mean_difference": 0.2, "ci_low": 0.02, "ci_high": 0.3},
                    "formal_pooling": "prohibited",
                }
            ),
            encoding="utf-8",
        )

    output = module.summarize_v3_robustness_cross_seed(root=root)

    table = pd.read_csv(root / "summary" / "cross_seed_effects.csv")
    assert output["formal_pooling"] == "prohibited"
    assert len(table) == 6
    assert "p_value" not in table.columns
    assert set(table["stratum"]) == {"primary", "high_pressure_m70_pivot"}


def test_protected_input_snapshot_is_immutable(tmp_path: Path) -> None:
    """运行前必须固定正式输入散列，且不能被后续步骤覆盖。"""
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    root = tmp_path / "research_artifacts" / "v3_robustness_20260810"
    protected = tmp_path / "formal.jsonl"
    protected.write_text('{"frozen":true}\n', encoding="utf-8")

    snapshot = module.write_protected_input_snapshot(
        root=root,
        protected_paths={"v3_formal": protected},
    )

    assert snapshot["protected_inputs"]["v3_formal"] == module._sha256(protected)
    with pytest.raises(FileExistsError):
        module.write_protected_input_snapshot(
            root=root,
            protected_paths={"v3_formal": protected},
        )


def test_robustness_run_uses_validation_selection_before_independent_test_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """删除独立流水线会让鲁棒性测试在未冻结 validation 选择时读取 test。"""
    module = importlib.import_module("paper1_active_req.experiment.v3_robustness")
    project_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "research_artifacts" / "v3_robustness_20260810"
    monkeypatch.setattr(module, "_ROBUSTNESS_TRAIN_SPEC_COUNT", 3)
    monkeypatch.setattr(module, "_ROBUSTNESS_VALIDATION_SPEC_COUNT", 3)
    monkeypatch.setattr(module, "_ROBUSTNESS_TEST_SPECS_PER_MECHANISM", 1)
    monkeypatch.setattr(module, "_ROBUSTNESS_PIVOT_SPECS_PER_MECHANISM", 1)
    monkeypatch.setattr(module, "_ROBUSTNESS_BUDGET", 2)

    manifest = module.run_v3_robustness_run(
        root=root,
        seed=20260813,
        schema_path=project_root / "configs" / "fields.yaml",
        relation_config_path=project_root / "configs" / "v3_relations.yaml",
        budget=2,
    )

    run_dir = root / "runs" / "seed-20260813"
    selection = json.loads((run_dir / "tuning" / "selected_lambda.json").read_text(encoding="utf-8"))
    frozen = json.loads((run_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "robustness-outcome.json").read_text(encoding="utf-8"))
    assert selection["selection_split"] == "validation"
    assert manifest.record_count == 90
    assert manifest.selected_lambda in {0.0, 0.05, 0.1, 0.2}
    assert outcome["status"] == "succeeded"
    assert outcome["record_count"] == 90
    assert set(manifest.policy_names) == set(module.V3_POLICY_NAMES)
    assert str(Path(v3_runner.__file__).resolve()) in frozen["source_hashes"]

    summary = module.summarize_v3_robustness_run(root=root, seed=20260813)
    assert summary["analysis_label"] == "independent_robustness_exploratory"
    assert summary["primary"]["n"] == 3
    assert summary["high_pressure_m70_pivot"]["n"] == 3
    assert (run_dir / "summary" / "robustness_effect.json").is_file()
    inference = pd.read_csv(run_dir / "summary" / "inference.csv")
    assert set(inference.columns) >= {
        "policy_name",
        "zero_filled_episode_precision_mean",
        "episode_with_inference_count",
        "inferred_total",
        "inferred_correct",
        "micro_precision",
        "continuous_reliable_inference_count",
    }
    assert inference.loc[0, "policy_name"] == "tr_mla"
    with pytest.raises(FileExistsError):
        module.run_v3_robustness_run(
            root=root,
            seed=20260813,
            schema_path=project_root / "configs" / "fields.yaml",
            relation_config_path=project_root / "configs" / "v3_relations.yaml",
            budget=2,
        )
