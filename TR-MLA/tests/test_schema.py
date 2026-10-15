from pathlib import Path

import pytest

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema


def test_schema_contains_exact_engineering_fields():
    schema = load_field_schema(Path("configs/fields.yaml"))

    assert len(schema) == 24
    assert schema["workpiece_mass_kg"].unit == "kg"
    assert schema["positioning_accuracy_mm"].criticality == 1.0
    assert schema["upstream_output_interface"].kind.value == "categorical"
    assert schema["power_supply"].queryable is False


def test_queryable_fields_include_hidden_conflicting_and_inferred_only():
    state = RequirementState(
        values={
            "workpiece_mass_kg": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "travel_distance_mm": FieldValue(
                value=1200.0, status=FieldStatus.OBSERVED
            ),
            "available_width_mm": FieldValue(
                value=200.0, status=FieldStatus.CONFLICTING
            ),
            "positioning_accuracy_mm": FieldValue(
                value=0.5, status=FieldStatus.INFERRED
            ),
            "workpiece_height_mm": FieldValue(
                value=None, status=FieldStatus.UNKNOWN_TO_USER
            ),
        }
    )

    assert state.queryable_field_ids() == [
        "available_width_mm",
        "positioning_accuracy_mm",
        "workpiece_mass_kg",
    ]


def test_apply_answer_does_not_mutate_previous_state():
    state = RequirementState(
        values={
            "workpiece_mass_kg": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            )
        }
    )

    updated = state.apply_answer(
        "workpiece_mass_kg",
        FieldValue(value=25.0, status=FieldStatus.OBSERVED, source="oracle"),
    )

    assert state.values["workpiece_mass_kg"].value is None
    assert updated.values["workpiece_mass_kg"].value == 25.0
    assert updated.asked_fields == ["workpiece_mass_kg"]


def test_schema_rejects_duplicate_field_ids(tmp_path: Path):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """
fields:
  - field_id: mass
    kind: continuous
    lower: 1
    upper: 10
    criticality: 1
    answer_cost: 1
  - field_id: mass
    kind: continuous
    lower: 2
    upper: 20
    criticality: 1
    answer_cost: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="field_id must be unique"):
        load_field_schema(duplicate)
