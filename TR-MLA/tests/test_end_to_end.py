import json
from pathlib import Path

from paper1_active_req.experiment.runner import (
    run_all_policies,
    summarize_results,
)
from paper1_active_req.generator.complete_specs import generate_dataset


def test_readme_contains_reproduction_contract():
    text = Path("README.md").read_text(encoding="utf-8")

    assert ".venv\\Scripts\\python.exe" in text
    assert "--training-specs" in text
    assert "--count 120" in text
    assert "不能访问 Oracle 保存的完整规格" in text
    assert "外部案例" in text


def test_end_to_end_smoke_experiment(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    result_file = tmp_path / "episodes.jsonl"
    summary_dir = tmp_path / "summary"

    generate_dataset(
        seed=17,
        count=6,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    run_all_policies(
        tasks_path=dataset_dir / "test_tasks.jsonl",
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=Path("configs/experiment.yaml"),
        output_path=result_file,
        seed=17,
        budget=2,
    )
    summarize_results(result_file, summary_dir)

    episodes = [
        json.loads(line)
        for line in result_file.read_text(encoding="utf-8").splitlines()
    ]
    assert {item["policy_name"] for item in episodes} == {
        "random",
        "fixed_questionnaire",
        "criticality_greedy",
        "uncertainty_afa",
        "constraint_range_only",
        "constraint_state_full",
    }
    assert all(len(item["episode"]["steps"]) <= 2 for item in episodes)
    assert (summary_dir / "metrics.csv").exists()
    assert (summary_dir / "pairwise_tests.csv").exists()
