import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper1_active_req.evaluation.statistics import (
    paired_bootstrap_mean_ci,
    paired_rank_biserial,
)
from paper1_active_req.experiment.runner import (
    run_constraint_state_ablation,
    run_all_policies,
    summarize_ablation_results,
    summarize_results,
)
from paper1_active_req.generator.complete_specs import generate_dataset


def test_rank_biserial_is_positive_for_positive_differences():
    assert paired_rank_biserial(np.array([0.2, 0.1, 0.4])) == 1.0


def test_bootstrap_interval_is_seeded_for_constant_difference():
    assert paired_bootstrap_mean_ci(
        np.array([0.2, 0.2, 0.2]), seed=20260731, resamples=100
    ) == pytest.approx((0.2, 0.2))


def test_runner_writes_resumable_records_and_summary(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=41,
        count=5,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    first_task = (
        dataset_dir / "test_tasks.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    one_task = tmp_path / "one_task.jsonl"
    one_task.write_text(f"{first_task}\n", encoding="utf-8")
    output = tmp_path / "episodes.jsonl"

    written = run_all_policies(
        tasks_path=one_task,
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        output_path=output,
        seed=41,
        budget=1,
    )
    repeated = run_all_policies(
        tasks_path=one_task,
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        output_path=output,
        seed=41,
        budget=1,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert written == 6
    assert repeated == 0
    assert {record["policy_name"] for record in records} == {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "constraint_state_full",
    }
    assert all(record["budget"] == 1 for record in records)

    summary_dir = tmp_path / "summary"
    summarize_results(output, summary_dir)

    metrics = pd.read_csv(summary_dir / "metrics.csv")
    assert set(metrics["policy_name"]) == {
        record["policy_name"] for record in records
    }
    assert (summary_dir / "pairwise_tests.csv").exists()
    comparisons = pd.read_csv(summary_dir / "pairwise_tests.csv")
    assert {
        "effect_size_rank_biserial",
        "mean_difference_ci_low",
        "mean_difference_ci_high",
    } <= set(comparisons.columns)


def test_stratified_summary_writes_separate_missing_rate_strata(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=59,
        count=12,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    tasks = [
        json.loads(line)
        for line in (dataset_dir / "test_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = [
        next(task for task in tasks if task["missing_rate"] == rate)
        for rate in (0.3, 0.7)
    ]
    tasks_path = tmp_path / "two_strata.jsonl"
    tasks_path.write_text(
        "\n".join(json.dumps(task) for task in selected) + "\n",
        encoding="utf-8",
    )
    episodes_path = tmp_path / "episodes.jsonl"
    run_all_policies(
        tasks_path=tasks_path,
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        output_path=episodes_path,
        seed=59,
        budget=1,
    )

    summary_dir = tmp_path / "summary"
    summarize_results(
        episodes_path,
        summary_dir,
        stratify_by_missing_rate=True,
    )

    stratified_metrics = pd.read_csv(
        summary_dir / "stratified_metrics.csv"
    )
    stratified_pairs = pd.read_csv(
        summary_dir / "stratified_pairwise_tests.csv"
    )
    expected_policies = {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "constraint_state_full",
    }
    assert set(stratified_metrics["missing_rate"]) == {0.3, 0.7}
    assert set(stratified_metrics["policy_name"]) == expected_policies
    assert set(stratified_pairs["missing_rate"]) == {0.3, 0.7}
    assert set(stratified_pairs["analysis_type"]) == {
        "post_hoc_diagnostic"
    }


def test_ablation_runner_writes_one_auditable_policy(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=71,
        count=5,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    first_task = (
        dataset_dir / "test_tasks.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    tasks_path = tmp_path / "one_task.jsonl"
    tasks_path.write_text(f"{first_task}\n", encoding="utf-8")
    selected_weights = tmp_path / "selected_weights.yaml"
    selected_weights.write_text(
        "selection_split: validation\n"
        "selected_weights:\n"
        "  domain_reduction: 0.5\n"
        "  criticality: 2.0\n"
        "  conflict: 2.0\n"
        "  graph_degree: 0.0\n"
        "  answer_cost: 0.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "ablate_conflict.jsonl"

    written = run_constraint_state_ablation(
        tasks_path=tasks_path,
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        selected_weights_path=selected_weights,
        variant_name="ablate_conflict",
        ablated_term="conflict",
        output_path=output,
        seed=71,
        budget=1,
    )

    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert written == 1
    assert record["policy_name"] == "constraint_state_ablate_conflict"
    assert record["analysis_metadata"]["analysis_type"] == (
        "post_hoc_diagnostic"
    )
    assert record["analysis_metadata"]["ablated_term"] == "conflict"
    assert record["analysis_metadata"]["resolved_weights"]["conflict"] == 0.0


def test_ablation_summary_pairs_full_and_variant_by_task(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=83,
        count=5,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    first_task = (
        dataset_dir / "test_tasks.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    tasks_path = tmp_path / "one_task.jsonl"
    tasks_path.write_text(f"{first_task}\n", encoding="utf-8")
    reference_path = tmp_path / "reference.jsonl"
    run_all_policies(
        tasks_path=tasks_path,
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        output_path=reference_path,
        seed=83,
        budget=1,
    )
    reference_records = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
    ]
    full_record = next(
        record
        for record in reference_records
        if record["policy_name"] == "constraint_state_full"
    )
    variant_record = json.loads(json.dumps(full_record))
    variant_record["policy_name"] = "constraint_state_ablate_conflict"
    variant_record["metrics"]["crr_at_8"] -= 0.25
    variant_record["analysis_metadata"] = {
        "analysis_type": "post_hoc_diagnostic",
        "variant_name": "ablate_conflict",
        "ablated_term": "conflict",
    }
    ablation_path = tmp_path / "ablation.jsonl"
    ablation_path.write_text(
        json.dumps(variant_record) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "ablation_summary.csv"

    summarize_ablation_results(
        reference_path,
        [ablation_path],
        output_path,
    )

    summary = pd.read_csv(output_path)
    assert list(summary["variant_name"]) == ["ablate_conflict"]
    assert summary.loc[0, "n"] == 1
    assert summary.loc[0, "analysis_type"] == "post_hoc_diagnostic"
    assert summary.loc[0, "mean_crr_difference_full_minus_ablation"] == (
        pytest.approx(0.25)
    )


def test_ablation_runner_refuses_an_output_with_active_writer_lock(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=97,
        count=5,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    first_task = (
        dataset_dir / "test_tasks.jsonl"
    ).read_text(encoding="utf-8").splitlines()[0]
    tasks_path = tmp_path / "one_task.jsonl"
    tasks_path.write_text(f"{first_task}\n", encoding="utf-8")
    selected_weights = tmp_path / "selected_weights.yaml"
    selected_weights.write_text(
        "selection_split: validation\n"
        "selected_weights:\n"
        "  domain_reduction: 0.5\n"
        "  criticality: 2.0\n"
        "  conflict: 2.0\n"
        "  graph_degree: 0.0\n"
        "  answer_cost: 0.5\n",
        encoding="utf-8",
    )
    output = tmp_path / "ablate_conflict.jsonl"
    lock_path = output.with_suffix(".jsonl.lock")
    lock_path.write_text("active", encoding="utf-8")

    with pytest.raises(FileExistsError, match="another writer"):
        run_constraint_state_ablation(
            tasks_path=tasks_path,
            training_specs_path=dataset_dir / "complete_specs.jsonl",
            schema_path=Path("configs/fields.yaml"),
            config_path=Path("configs/experiment.yaml"),
            selected_weights_path=selected_weights,
            variant_name="ablate_conflict",
            ablated_term="conflict",
            output_path=output,
            seed=97,
            budget=1,
        )

    assert not output.exists()
    assert lock_path.exists()
