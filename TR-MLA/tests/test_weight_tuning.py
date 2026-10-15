from pathlib import Path

import yaml

from paper1_active_req.experiment.tuning import (
    SuccessiveHalvingRecord,
    WeightSearchRecord,
    iter_weight_grid,
    run_successive_halving_weight_search,
    run_weight_search,
    sample_weight_candidates,
    select_stage_one_tasks,
    select_best_weight,
)
from paper1_active_req.experiment.runner import resolve_utility_weights
from paper1_active_req.generator.complete_specs import generate_dataset
from paper1_active_req.policies.constraint_state import UtilityWeights


def test_weight_grid_and_tie_breaking_are_deterministic():
    grid = {
        "domain_reduction": [1.0],
        "criticality": [1.0],
        "conflict": [1.0],
        "graph_degree": [0.0],
        "answer_cost": [0.25, 1.0],
    }

    candidates = iter_weight_grid(grid)
    records = [
        WeightSearchRecord(
            weights=candidates[0],
            mean_crr_at_8=0.8,
            mean_answer_cost=2.0,
            mean_solver_calls=10.0,
            n=3,
        ),
        WeightSearchRecord(
            weights=candidates[1],
            mean_crr_at_8=0.8,
            mean_answer_cost=1.5,
            mean_solver_calls=12.0,
            n=3,
        ),
    ]

    assert len(candidates) == 2
    assert select_best_weight(records).weights.answer_cost == 1.0


def test_candidate_sampling_is_seeded_unique_and_keeps_default():
    grid = {
        "domain_reduction": [0.5, 1.0],
        "criticality": [0.5, 1.0],
        "conflict": [1.0],
        "graph_degree": [0.0],
        "answer_cost": [0.25, 0.5],
    }
    default = UtilityWeights(
        domain_reduction=1.0,
        criticality=1.0,
        conflict=1.0,
        graph_degree=0.0,
        answer_cost=0.5,
    )

    first = sample_weight_candidates(
        grid, default, seed=7, count=4
    )
    second = sample_weight_candidates(
        grid, default, seed=7, count=4
    )

    assert first == second
    assert first[0] == default
    assert len(first) == 4
    assert len({tuple(item.model_dump().values()) for item in first}) == 4


def test_stage_one_task_selection_keeps_all_missing_rates_per_spec(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=52,
        count=15,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    from paper1_active_req.experiment.runner import _load_jsonl
    from paper1_active_req.generator.complete_specs import RequirementTask

    validation_tasks = _load_jsonl(
        dataset_dir / "validation_tasks.jsonl", RequirementTask
    )
    selected = select_stage_one_tasks(
        validation_tasks, seed=7, spec_count=2
    )

    assert len(selected) == 6
    assert selected == select_stage_one_tasks(
        validation_tasks, seed=7, spec_count=2
    )
    for spec_id in {task.spec_id for task in selected}:
        assert {
            task.missing_rate
            for task in selected
            if task.spec_id == spec_id
        } == {0.3, 0.5, 0.7}


def test_weight_search_uses_validation_split_and_writes_selection(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=51,
        count=5,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    config = {
        "max_numeric_outcomes": 2,
        "weight_grid": {
            "domain_reduction": [0.0],
            "criticality": [1.0],
            "conflict": [1.0],
            "graph_degree": [0.0],
            "answer_cost": [0.0, 1.0],
        },
    }
    config_path = tmp_path / "tuning.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    records_path = tmp_path / "weight_records.jsonl"
    selected_path = tmp_path / "selected_weights.yaml"

    best = run_weight_search(
        validation_tasks_path=dataset_dir / "validation_tasks.jsonl",
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=config_path,
        records_path=records_path,
        selected_path=selected_path,
        seed=51,
        budget=1,
    )

    lines = records_path.read_text(encoding="utf-8").splitlines()
    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    assert len(lines) == 2
    assert best.n == 3
    assert UtilityWeights.model_validate(
        selected["selected_weights"]
    ) == best.weights
    assert resolve_utility_weights(config, selected_path) == best.weights
    assert selected["selection_split"] == "validation"


def test_successive_halving_uses_validation_and_writes_two_stages(
    tmp_path: Path,
):
    dataset_dir = tmp_path / "dataset"
    generate_dataset(
        seed=53,
        count=10,
        output_dir=dataset_dir,
        schema_path=Path("configs/fields.yaml"),
    )
    config = {
        "max_numeric_outcomes": 2,
        "default_weights": {
            "domain_reduction": 1.0,
            "criticality": 1.0,
            "conflict": 1.0,
            "graph_degree": 0.0,
            "answer_cost": 0.5,
        },
        "weight_grid": {
            "domain_reduction": [0.5, 1.0],
            "criticality": [0.5, 1.0],
            "conflict": [1.0],
            "graph_degree": [0.0],
            "answer_cost": [0.25, 0.5],
        },
    }
    config_path = tmp_path / "successive_halving.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    records_path = tmp_path / "staged_records.jsonl"
    selected_path = tmp_path / "selected_weights.yaml"

    best = run_successive_halving_weight_search(
        validation_tasks_path=dataset_dir / "validation_tasks.jsonl",
        training_specs_path=dataset_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        config_path=config_path,
        records_path=records_path,
        selected_path=selected_path,
        seed=53,
        budget=1,
        candidate_count=4,
        stage_one_spec_count=1,
        finalists=2,
    )

    records = [
        SuccessiveHalvingRecord.model_validate_json(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    assert {record.stage for record in records} == {"stage1", "stage2"}
    assert all(record.task_split == "validation" for record in records)
    assert all(record.n > 0 for record in records)
    assert best.n == len(
        (dataset_dir / "validation_tasks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert selected["selection_split"] == "validation"
    assert selected["selected_weights"] == best.weights.model_dump()
    assert selected["candidate_count"] == 4
    assert selected["finalist_count"] in {2, 3}
    assert selected["selected_candidate_id"].startswith("candidate-")
    assert selected["config_hash"] == records[0].config_hash
