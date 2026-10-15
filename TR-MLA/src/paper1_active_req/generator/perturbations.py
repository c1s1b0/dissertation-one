from __future__ import annotations

import numpy as np

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)

from .complete_specs import CompleteSpec, RequirementTask


def mask_spec(
    spec: CompleteSpec,
    missing_rate: float,
    rng: np.random.Generator,
    queryable_field_ids: set[str] | None = None,
) -> RequirementTask:
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between 0 and 1")
    if queryable_field_ids is None:
        queryable_field_ids = set(spec.values) - {"power_supply"}
    candidates = sorted(queryable_field_ids)
    hidden_count = round(len(candidates) * missing_rate)
    hidden_ids = set(
        rng.choice(candidates, size=hidden_count, replace=False).tolist()
    )

    initial_values = {
        field_id: field_value.model_copy(deep=True)
        for field_id, field_value in spec.values.items()
    }
    for field_id in hidden_ids:
        initial_values[field_id] = FieldValue(
            value=None,
            status=FieldStatus.HIDDEN,
            source="mask",
        )
    ground_truth = {
        field_id: field_value.model_copy(deep=True)
        for field_id, field_value in spec.values.items()
    }
    rate_label = int(round(missing_rate * 100))
    return RequirementTask(
        task_id=f"{spec.spec_id}__m{rate_label}",
        spec_id=spec.spec_id,
        split=spec.split,
        missing_rate=missing_rate,
        initial_state=RequirementState(values=initial_values),
        ground_truth=ground_truth,
    )


def inject_conflict(
    task: RequirementTask,
    conflict_type: str,
    rng: np.random.Generator,
) -> RequirementTask:
    del rng
    values = {
        field_id: field_value.model_copy(deep=True)
        for field_id, field_value in task.initial_state.values.items()
    }
    truth = task.ground_truth

    def reveal(field_id: str) -> None:
        values[field_id] = truth[field_id].model_copy(
            update={"status": FieldStatus.OBSERVED}
        )

    if conflict_type == "input_interface":
        reveal("upstream_output_interface")
        original = str(truth["upstream_output_interface"].value)
        alternatives = [
            value
            for value in ("belt", "roller", "pallet")
            if value != original
        ]
        field_id = "station_input_interface"
        values[field_id] = FieldValue(
            value=alternatives[0],
            status=FieldStatus.CONFLICTING,
            source="conflict_injection",
        )
    elif conflict_type == "available_width":
        reveal("workpiece_width_mm")
        reveal("side_clearance_mm")
        field_id = "available_width_mm"
        values[field_id] = FieldValue(
            value=200.0,
            status=FieldStatus.CONFLICTING,
            source="conflict_injection",
        )
    elif conflict_type == "load_capacity":
        reveal("workpiece_mass_kg")
        reveal("safety_factor")
        required = (
            float(truth["workpiece_mass_kg"].value)
            * float(truth["safety_factor"].value)
        )
        field_id = "load_capacity_kg"
        values[field_id] = FieldValue(
            value=max(1.0, required * 0.5),
            status=FieldStatus.CONFLICTING,
            source="conflict_injection",
        )
    else:
        raise ValueError(f"unknown conflict_type: {conflict_type}")

    return task.model_copy(
        update={
            "initial_state": RequirementState(values=values),
            "injected_conflict_type": conflict_type,
            "injected_conflict_field": field_id,
        }
    )
