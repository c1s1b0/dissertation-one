import json
from pathlib import Path

import numpy as np

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.v2_runner import (
    run_v2_selected_experiment,
    run_v2_validation_tuning,
    summarize_v2_results,
)
from paper1_active_req.generator.complete_specs import generate_complete_spec
from paper1_active_req.generator.perturbations import mask_spec
from paper1_active_req.generator.v2_pivotal_specs import (
    V2AuditMetadata,
    V2RequirementTask,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _write_jsonl(path: Path, models: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(model.model_dump_json() for model in models) + "\n",
        encoding="utf-8",
    )


def _v2_task(spec, rng: np.random.Generator) -> V2RequirementTask:
    base = mask_spec(spec, missing_rate=0.3, rng=rng)
    return V2RequirementTask.model_validate(
        {
            **base.model_dump(),
            "audit": V2AuditMetadata(
                module_id="load_space",
                pivot_field_id=None,
                affected_critical_field_ids=(),
                mechanism="interval_reduction",
            ).model_dump(),
        }
    )


def test_v2_smoke_runs_validation_selection_then_test_and_summary(tmp_path: Path):
    """防止四个入口单独可用却无法组成无泄漏实验闭环。"""
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(29)
    train_spec = generate_complete_spec(rng, schema, engine, "v2-spec-train")
    validation_spec = generate_complete_spec(rng, schema, engine, "v2-spec-validation")
    test_spec = generate_complete_spec(rng, schema, engine, "v2-spec-test")
    train_spec.split = "train"
    validation_spec.split = "validation"
    test_spec.split = "test"
    data_dir = tmp_path / "v2" / "data"
    _write_jsonl(data_dir / "complete_specs.jsonl", [train_spec, validation_spec, test_spec])
    _write_jsonl(data_dir / "validation_tasks.jsonl", [_v2_task(validation_spec, rng)])
    _write_jsonl(data_dir / "test_tasks.jsonl", [_v2_task(test_spec, rng)])
    tuning_dir = tmp_path / "v2" / "tuning"

    selected = run_v2_validation_tuning(
        validation_tasks_path=data_dir / "validation_tasks.jsonl",
        training_specs_path=data_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        output_dir=tuning_dir,
        seed=29,
        budget=1,
    )
    episodes_path = tmp_path / "v2" / "smoke" / "episodes.jsonl"
    written = run_v2_selected_experiment(
        tasks_path=data_dir / "test_tasks.jsonl",
        training_specs_path=data_dir / "complete_specs.jsonl",
        schema_path=Path("configs/fields.yaml"),
        selected_lambdas_path=tuning_dir / "selected_lambdas.yaml",
        output_path=episodes_path,
        seed=29,
        budget=1,
    )
    summary_dir = tmp_path / "v2" / "smoke" / "summary"
    summarize_v2_results(episodes_path, summary_dir)

    assert selected["selection_split"] == "validation"
    assert written == 8
    assert len(episodes_path.read_text(encoding="utf-8").splitlines()) == 8
    assert set(
        json.loads(line)["split"]
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
    ) == {"test"}
    assert (summary_dir / "metrics.csv").exists()
    assert (summary_dir / "pairwise_tests.csv").exists()
