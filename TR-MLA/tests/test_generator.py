from pathlib import Path

import numpy as np

from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import (
    generate_complete_spec,
    generate_dataset,
)
from paper1_active_req.solver.engine import ConstraintEngine


def test_generated_complete_specs_are_satisfiable():
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(20260731)

    specs = [
        generate_complete_spec(rng, schema, engine, f"spec-{index:03d}")
        for index in range(20)
    ]

    assert len({spec.spec_id for spec in specs}) == 20
    assert all(
        engine.check(spec.to_requirement_state()).satisfiable
        for spec in specs
    )


def test_dataset_split_happens_before_sparse_variants(tmp_path: Path):
    output = tmp_path / "dataset"

    manifest = generate_dataset(
        seed=17,
        count=12,
        output_dir=output,
        schema_path=Path("configs/fields.yaml"),
    )

    assert manifest.complete_spec_count == 12
    assert manifest.task_count == 36
    assert manifest.split_spec_counts == {
        "train": 7,
        "validation": 2,
        "test": 3,
    }
    assert set(manifest.train_spec_ids).isdisjoint(manifest.test_spec_ids)
    assert (output / "complete_specs.jsonl").exists()
    assert (output / "test_tasks.jsonl").exists()
