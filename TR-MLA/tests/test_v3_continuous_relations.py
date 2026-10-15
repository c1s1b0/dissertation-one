import math
from pathlib import Path

import pytest

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import (
    RelationFamily,
    RelationSpec,
    load_v3_relation_config,
)
from paper1_active_req.relations.continuous import ContinuousRelationFactor


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED, source="fixture")


def _hidden() -> FieldValue:
    return FieldValue(value=None, status=FieldStatus.HIDDEN, source="fixture")


def _specs(rows: list[dict[str, object]]) -> list[CompleteSpec]:
    return [
        CompleteSpec(
            spec_id=f"training-{index:03d}",
            values={field_id: _observed(value) for field_id, value in row.items()},
        )
        for index, row in enumerate(rows)
    ]


def _state(**values: object) -> RequirementState:
    return RequirementState(
        values={
            field_id: _hidden() if value is None else _observed(value)
            for field_id, value in values.items()
        }
    )


def clearance_relation() -> RelationSpec:
    return RelationSpec(
        relation_id="width_clearance",
        family=RelationFamily.AFFINE,
        target_field_id="available_width_mm",
        parent_field_ids=("workpiece_width_mm",),
    )


def training_clearance_specs() -> list[CompleteSpec]:
    return _specs(
        [
            {"workpiece_width_mm": 300.0, "available_width_mm": 382.0},
            {"workpiece_width_mm": 310.0, "available_width_mm": 398.0},
            {"workpiece_width_mm": 320.0, "available_width_mm": 402.0},
            {"workpiece_width_mm": 330.0, "available_width_mm": 418.0},
        ]
    )


def test_affine_factor_generalizes_to_continuous_values_absent_from_training():
    factor = ContinuousRelationFactor.fit(training_clearance_specs(), clearance_relation())

    posterior = factor.posterior(_state(workpiece_width_mm=317.25))
    values = [float(branch.value) for branch in posterior.branches]

    assert len(values) == 3
    assert values[0] < values[1] < values[2]
    assert min(values) < 403.0 < max(values)


def test_affine_factor_exposes_residual_interval_wider_than_three_branches():
    factor = ContinuousRelationFactor.fit(training_clearance_specs(), clearance_relation())

    posterior = factor.posterior(_state(workpiece_width_mm=317.25))

    assert posterior.interval_low < float(posterior.branches[0].value)
    assert posterior.interval_high > float(posterior.branches[-1].value)
    assert sum(branch.probability for branch in posterior.branches) == pytest.approx(1.0)


def test_continuous_factor_keeps_95_percent_interval_separate_from_three_branches():
    factor = ContinuousRelationFactor.fit(
        _specs(
            [
                {"workpiece_width_mm": 0.0, "available_width_mm": 90.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 100.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 100.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 100.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 110.0},
            ]
        ),
        clearance_relation(),
    )

    posterior = factor.posterior(_state(workpiece_width_mm=0.0))

    assert float(posterior.branches[-1].value) - float(posterior.branches[0].value) == pytest.approx(12.0)
    assert posterior.interval_high - posterior.interval_low == pytest.approx(18.0)


def test_continuous_factor_uses_asymmetric_empirical_residual_branch_masses():
    factor = ContinuousRelationFactor.fit(
        _specs(
            [
                {"workpiece_width_mm": 0.0, "available_width_mm": 0.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 0.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 0.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 0.0},
                {"workpiece_width_mm": 0.0, "available_width_mm": 10.0},
            ]
        ),
        clearance_relation(),
    )

    posterior = factor.posterior(_state(workpiece_width_mm=0.0))

    assert tuple(branch.probability for branch in posterior.branches) == pytest.approx(
        (5 / 8, 1 / 8, 2 / 8)
    )
    assert posterior.confidence == pytest.approx(5 / 8)


def test_ratio_and_inverse_time_reject_unsafe_runtime_parent_values():
    for family, target_field_id, parent_field_ids in (
        (RelationFamily.RATIO, "ratio_target", ("numerator", "denominator")),
        (
            RelationFamily.INVERSE_TIME,
            "transfer_time_s",
            ("travel_distance_mm", "conveyor_speed_mm_s"),
        ),
    ):
        relation = RelationSpec(
            relation_id=f"{family}_runtime",
            family=family,
            target_field_id=target_field_id,
            parent_field_ids=parent_field_ids,
        )
        rows = [
            dict(zip(parent_field_ids, (2.0, 1.0), strict=True), **{target_field_id: 3.0}),
            dict(zip(parent_field_ids, (4.0, 2.0), strict=True), **{target_field_id: 5.0}),
            dict(zip(parent_field_ids, (6.0, 3.0), strict=True), **{target_field_id: 7.0}),
        ]
        factor = ContinuousRelationFactor.fit(_specs(rows), relation)

        with pytest.raises(ValueError, match="zero denominator"):
            factor.posterior(
                _state(**dict(zip(parent_field_ids, (2.0, 0.0), strict=True)) )
            )
        with pytest.raises(ValueError, match="finite"):
            factor.posterior(
                _state(**dict(zip(parent_field_ids, (math.inf, 1.0), strict=True)) )
            )


def test_inverse_time_rejects_unsafe_training_denominator_and_uses_frozen_direction():
    relation = load_v3_relation_config(Path("configs/v3_relations.yaml")).relations[6]
    good_specs = _specs(
        [
            {"travel_distance_mm": 100.0, "conveyor_speed_mm_s": 50.0, "transfer_time_s": 3.0},
            {"travel_distance_mm": 300.0, "conveyor_speed_mm_s": 75.0, "transfer_time_s": 5.0},
            {"travel_distance_mm": 600.0, "conveyor_speed_mm_s": 100.0, "transfer_time_s": 7.0},
        ]
    )

    posterior = ContinuousRelationFactor.fit(good_specs, relation).posterior(
        _state(travel_distance_mm=200.0, conveyor_speed_mm_s=50.0)
    )

    assert relation.relation_id == "travel_speed_time"
    assert float(posterior.branches[1].value) == pytest.approx(5.0)
    invalid_specs = good_specs[:2] + [
        CompleteSpec(
            spec_id="zero-speed",
            values={
                "travel_distance_mm": _observed(200.0),
                "conveyor_speed_mm_s": _observed(0.0),
                "transfer_time_s": _observed(5.0),
            },
        )
    ]
    with pytest.raises(ValueError, match="zero denominator"):
        ContinuousRelationFactor.fit(invalid_specs, relation)


def test_ratio_factor_transforms_two_parents_before_fitting():
    relation = RelationSpec(
        relation_id="ratio_relation",
        family=RelationFamily.RATIO,
        target_field_id="ratio_target",
        parent_field_ids=("numerator", "denominator"),
    )
    factor = ContinuousRelationFactor.fit(
        _specs(
            [
                {"numerator": 2.0, "denominator": 1.0, "ratio_target": 5.0},
                {"numerator": 6.0, "denominator": 2.0, "ratio_target": 7.0},
                {"numerator": 12.0, "denominator": 3.0, "ratio_target": 9.0},
            ]
        ),
        relation,
    )

    posterior = factor.posterior(_state(numerator=9.0, denominator=3.0))

    assert float(posterior.branches[1].value) == pytest.approx(7.0)


def test_inverse_time_factor_transforms_distance_and_speed_before_fitting():
    relation = RelationSpec(
        relation_id="travel_speed_time",
        family=RelationFamily.INVERSE_TIME,
        target_field_id="transfer_time_s",
        parent_field_ids=("travel_distance_mm", "conveyor_speed_mm_s"),
    )
    factor = ContinuousRelationFactor.fit(
        _specs(
            [
                {"travel_distance_mm": 100.0, "conveyor_speed_mm_s": 50.0, "transfer_time_s": 3.0},
                {"travel_distance_mm": 300.0, "conveyor_speed_mm_s": 75.0, "transfer_time_s": 5.0},
                {"travel_distance_mm": 600.0, "conveyor_speed_mm_s": 100.0, "transfer_time_s": 7.0},
            ]
        ),
        relation,
    )

    posterior = factor.posterior(
        _state(travel_distance_mm=200.0, conveyor_speed_mm_s=50.0)
    )

    assert float(posterior.branches[1].value) == pytest.approx(5.0)


def test_gated_affine_fits_each_supported_gate_and_uses_global_fallback():
    relation = RelationSpec(
        relation_id="fragile_positioning",
        family=RelationFamily.GATED_AFFINE,
        target_field_id="positioning_accuracy_mm",
        parent_field_ids=("repeatability_mm",),
        gate_field_id="workpiece_fragile",
        minimum_support=3,
    )
    factor = ContinuousRelationFactor.fit(
        _specs(
            [
                {"repeatability_mm": 1.0, "workpiece_fragile": False, "positioning_accuracy_mm": 11.0},
                {"repeatability_mm": 2.0, "workpiece_fragile": False, "positioning_accuracy_mm": 12.0},
                {"repeatability_mm": 3.0, "workpiece_fragile": False, "positioning_accuracy_mm": 13.0},
                {"repeatability_mm": 1.0, "workpiece_fragile": True, "positioning_accuracy_mm": 21.0},
                {"repeatability_mm": 2.0, "workpiece_fragile": True, "positioning_accuracy_mm": 23.0},
                {"repeatability_mm": 3.0, "workpiece_fragile": True, "positioning_accuracy_mm": 25.0},
            ]
        ),
        relation,
    )

    fragile = factor.posterior(_state(repeatability_mm=4.0, workpiece_fragile=True))
    missing_gate = factor.posterior(_state(repeatability_mm=4.0, workpiece_fragile=None))

    assert float(fragile.branches[1].value) == pytest.approx(27.0)
    assert float(missing_gate.branches[1].value) == pytest.approx(20.5)


@pytest.mark.parametrize(
    ("rows", "state", "match"),
    [
        (
            [
                {"numerator": 1.0, "denominator": 0.0, "ratio_target": 1.0},
                {"numerator": 2.0, "denominator": 1.0, "ratio_target": 2.0},
                {"numerator": 3.0, "denominator": 1.0, "ratio_target": 3.0},
            ],
            None,
            "zero denominator",
        ),
        (
            [
                {"numerator": 1.0, "denominator": 1.0, "ratio_target": math.inf},
                {"numerator": 2.0, "denominator": 1.0, "ratio_target": 2.0},
                {"numerator": 3.0, "denominator": 1.0, "ratio_target": 3.0},
            ],
            None,
            "finite",
        ),
    ],
)
def test_ratio_factor_rejects_unsafe_training_transformations(
    rows: list[dict[str, object]], state: object, match: str
):
    del state
    relation = RelationSpec(
        relation_id="ratio_relation",
        family=RelationFamily.RATIO,
        target_field_id="ratio_target",
        parent_field_ids=("numerator", "denominator"),
    )

    with pytest.raises(ValueError, match=match):
        ContinuousRelationFactor.fit(_specs(rows), relation)


def test_continuous_factor_rejects_insufficient_support():
    relation = clearance_relation().model_copy(update={"minimum_support": 4})

    with pytest.raises(ValueError, match="minimum support"):
        ContinuousRelationFactor.fit(training_clearance_specs()[:3], relation)
