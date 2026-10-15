from pathlib import Path

import numpy as np
import pytest

from paper1_active_req.domain.models import FieldStatus
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import generate_complete_spec
from paper1_active_req.generator.perturbations import (
    inject_conflict,
    mask_spec,
)
from paper1_active_req.solver.engine import ConstraintEngine


@pytest.fixture
def complete_spec():
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(11)
    return generate_complete_spec(rng, schema, engine, "spec-test")


def test_masking_never_changes_ground_truth(complete_spec):
    rng = np.random.default_rng(12)

    task = mask_spec(complete_spec, missing_rate=0.5, rng=rng)

    assert task.ground_truth == complete_spec.values
    assert task.ground_truth is not complete_spec.values
    hidden = [
        value
        for value in task.initial_state.values.values()
        if value.status == FieldStatus.HIDDEN
    ]
    assert len(hidden) == 12
    assert task.initial_state.values["power_supply"].status == (
        FieldStatus.OBSERVED
    )


@pytest.mark.parametrize(
    ("conflict_type", "expected_rule"),
    [
        ("input_interface", "input_interface_match"),
        ("available_width", "available_width_clearance"),
        ("load_capacity", "load_capacity_safety"),
    ],
)
def test_injected_conflict_is_detected(
    complete_spec,
    conflict_type: str,
    expected_rule: str,
):
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(13)
    task = mask_spec(complete_spec, missing_rate=0.3, rng=rng)

    task = inject_conflict(task, conflict_type=conflict_type, rng=rng)
    result = engine.check(task.initial_state)

    assert result.satisfiable is False
    assert expected_rule in result.unsat_core
    assert task.injected_conflict_field is not None
