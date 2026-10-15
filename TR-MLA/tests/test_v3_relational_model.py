import pytest

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldKind,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import (
    RelationFamily,
    RelationSpec,
    V3RelationConfig,
)
from paper1_active_req.relations.model import TypedRelationalModel


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED, source="fixture")


def _hidden() -> FieldValue:
    return FieldValue(value=None, status=FieldStatus.HIDDEN, source="fixture")


def _schema() -> dict[str, FieldDefinition]:
    return {
        "workpiece_width_mm": FieldDefinition(field_id="workpiece_width_mm", kind=FieldKind.CONTINUOUS, lower=1.0, upper=1000.0, criticality=0.5, answer_cost=0.2),
        "available_width_mm": FieldDefinition(field_id="available_width_mm", kind=FieldKind.CONTINUOUS, lower=1.0, upper=3000.0, criticality=1.0, answer_cost=0.2),
        "repeatability_mm": FieldDefinition(field_id="repeatability_mm", kind=FieldKind.CONTINUOUS, lower=0.01, upper=10.0, criticality=0.5, answer_cost=0.2),
        "positioning_accuracy_mm": FieldDefinition(field_id="positioning_accuracy_mm", kind=FieldKind.CONTINUOUS, lower=0.01, upper=100.0, criticality=1.0, answer_cost=0.2),
        "workpiece_fragile": FieldDefinition(field_id="workpiece_fragile", kind=FieldKind.BOOLEAN, criticality=0.5, answer_cost=0.2),
        "unrelated": FieldDefinition(field_id="unrelated", kind=FieldKind.CATEGORICAL, categories=("a", "b"), criticality=0.5, answer_cost=0.2),
    }


def _config() -> V3RelationConfig:
    return V3RelationConfig(
        field_schema_path="unused-by-fitted-model.yaml",
        confidence_threshold=0.95,
        continuous_quantiles=(0.1, 0.5, 0.9),
        continuous_interval_quantiles=(0.025, 0.975),
        engineering_tolerances={"available_width_mm": 10.0, "positioning_accuracy_mm": 0.05},
        relations=(
            RelationSpec(relation_id="width_clearance", family=RelationFamily.AFFINE, target_field_id="available_width_mm", parent_field_ids=("workpiece_width_mm",)),
            RelationSpec(relation_id="fragile_positioning", family=RelationFamily.GATED_AFFINE, target_field_id="positioning_accuracy_mm", parent_field_ids=("repeatability_mm",), gate_field_id="workpiece_fragile"),
        ),
    )


def _training_specs() -> list[CompleteSpec]:
    rows = []
    for index, (width, residual, fragile) in enumerate(
        ((200.0, -2.0, True), (250.0, 2.0, True), (300.0, -2.0, True), (350.0, 2.0, False), (400.0, -2.0, False), (450.0, 2.0, False))
    ):
        repeatability = float(index + 1)
        positioning = (2.0 if fragile else 1.0) * repeatability + 10.0 + (-0.1 if index % 2 == 0 else 0.1)
        rows.append(
            CompleteSpec(
                spec_id=f"training-{index:03d}",
                values={
                    "workpiece_width_mm": _observed(width),
                    "available_width_mm": _observed(width + 85.0 + residual),
                    "repeatability_mm": _observed(repeatability),
                    "positioning_accuracy_mm": _observed(positioning),
                    "workpiece_fragile": _observed(fragile),
                    "unrelated": _observed("a" if index < 4 else "b"),
                },
            )
        )
    return rows


def fitted_typed_model() -> TypedRelationalModel:
    return TypedRelationalModel.fit(_training_specs(), _schema(), _config())


def state_with_known_clearance_parents() -> RequirementState:
    return RequirementState(
        values={
            "workpiece_width_mm": _observed(317.25),
            "available_width_mm": _hidden(),
            "repeatability_mm": _observed(4.0),
            "positioning_accuracy_mm": _hidden(),
            "workpiece_fragile": _observed(True),
            "unrelated": _hidden(),
        }
    )


def test_relational_model_resolves_only_when_frozen_threshold_or_tolerance_is_met():
    model = fitted_typed_model()

    resolved = model.reliably_resolved(state_with_known_clearance_parents())

    assert "available_width_mm" in resolved
    assert "positioning_accuracy_mm" not in resolved


def test_relational_model_uses_95_percent_interval_not_80_percent_branch_span():
    schema = {
        field_id: FieldDefinition(
            field_id=field_id,
            kind=FieldKind.CONTINUOUS,
            lower=0.0,
            upper=1000.0,
            criticality=1.0,
            answer_cost=0.2,
        )
        for field_id in ("parent", "target")
    }
    specs = [
        CompleteSpec(
            spec_id=f"wide-{index}",
            values={"parent": _observed(0.0), "target": _observed(target)},
        )
        for index, target in enumerate((90.0, 100.0, 100.0, 100.0, 110.0))
    ]
    config = V3RelationConfig(
        field_schema_path="unused.yaml",
        confidence_threshold=0.95,
        continuous_quantiles=(0.1, 0.5, 0.9),
        continuous_interval_quantiles=(0.025, 0.975),
        engineering_tolerances={"target": 15.0},
        relations=(
            RelationSpec(
                relation_id="wide_residuals",
                family=RelationFamily.AFFINE,
                target_field_id="target",
                parent_field_ids=("parent",),
            ),
        ),
    )
    state = RequirementState(values={"parent": _observed(0.0), "target": _hidden()})

    resolved = TypedRelationalModel.fit(specs, schema, config).reliably_resolved(state)

    assert "target" not in resolved


def test_relational_model_falls_back_to_training_marginal_without_applicable_relation():
    model = fitted_typed_model()

    posterior = model.posterior(state_with_known_clearance_parents(), "unrelated")

    assert tuple(branch.value for branch in posterior.branches) == ("a", "b")
    assert tuple(branch.probability for branch in posterior.branches) == pytest.approx((4 / 6, 2 / 6))
    assert posterior.factor_ids == ()


def test_relational_model_combines_discrete_factors_by_normalized_product():
    schema = {
        field_id: FieldDefinition(
            field_id=field_id,
            kind=FieldKind.CATEGORICAL,
            categories=("belt", "roller", "pallet"),
            criticality=0.5,
            answer_cost=0.2,
        )
        for field_id in (
            "station_input_interface",
            "upstream_output_interface",
            "station_output_interface",
        )
    }
    specs = [
        CompleteSpec(
            spec_id=f"joint-{index}",
            values={
                "station_input_interface": _observed(input_interface),
                "upstream_output_interface": _observed(upstream_interface),
                "station_output_interface": _observed("roller"),
            },
        )
        for index, (input_interface, upstream_interface) in enumerate(
            [("roller", "roller")] * 4
            + [("roller", "belt"), ("belt", "roller")]
        )
    ]
    config = V3RelationConfig(
        field_schema_path="unused.yaml",
        confidence_threshold=0.95,
        continuous_quantiles=(0.1, 0.5, 0.9),
        continuous_interval_quantiles=(0.025, 0.975),
        engineering_tolerances={},
        relations=(
            RelationSpec(
                relation_id="input_output",
                family=RelationFamily.DISCRETE_CPT,
                target_field_id="station_output_interface",
                parent_field_ids=("station_input_interface",),
                minimum_support=1,
            ),
            RelationSpec(
                relation_id="upstream_output",
                family=RelationFamily.DISCRETE_CPT,
                target_field_id="station_output_interface",
                parent_field_ids=("upstream_output_interface",),
                minimum_support=1,
            ),
        ),
    )
    state = RequirementState(
        values={
            "station_input_interface": _observed("roller"),
            "upstream_output_interface": _observed("roller"),
            "station_output_interface": _hidden(),
        }
    )

    posterior = TypedRelationalModel.fit(specs, schema, config).posterior(
        state, "station_output_interface"
    )

    assert posterior.branches[0].value == "roller"
    assert posterior.branches[0].probability == pytest.approx(18 / 19)
    assert posterior.factor_ids == ("input_output", "upstream_output")


def test_relational_model_combines_continuous_factors_by_inverse_variance():
    schema = {
        field_id: FieldDefinition(
            field_id=field_id,
            kind=FieldKind.CONTINUOUS,
            lower=0.0,
            upper=100.0,
            criticality=0.5,
            answer_cost=0.2,
        )
        for field_id in ("parent_a", "parent_b", "target")
    }
    specs = [
        CompleteSpec(
            spec_id=f"continuous-{index}",
            values={
                "parent_a": _observed(parent_a),
                "parent_b": _observed(parent_b),
                "target": _observed(target),
            },
        )
        for index, (parent_a, parent_b, target) in enumerate(
            ((0.0, 1.0, 10.0), (2.0, 2.0, 12.0), (2.0, 3.0, 12.0), (6.0, 4.0, 16.0))
        )
    ]
    config = V3RelationConfig(
        field_schema_path="unused.yaml",
        confidence_threshold=0.95,
        continuous_quantiles=(0.1, 0.5, 0.9),
        continuous_interval_quantiles=(0.025, 0.975),
        engineering_tolerances={"target": 0.1},
        relations=(
            RelationSpec(
                relation_id="exact", family=RelationFamily.AFFINE, target_field_id="target", parent_field_ids=("parent_a",)
            ),
            RelationSpec(
                relation_id="noisy", family=RelationFamily.AFFINE, target_field_id="target", parent_field_ids=("parent_b",)
            ),
        ),
    )
    state = RequirementState(
        values={"parent_a": _observed(3.0), "parent_b": _observed(2.5), "target": _hidden()}
    )

    posterior = TypedRelationalModel.fit(specs, schema, config).posterior(state, "target")

    assert float(posterior.branches[1].value) == pytest.approx(13.0)
    assert posterior.factor_ids == ("exact", "noisy")


def test_continuous_aggregation_preserves_empirical_branch_masses():
    schema = {
        field_id: FieldDefinition(
            field_id=field_id,
            kind=FieldKind.CONTINUOUS,
            lower=0.0,
            upper=100.0,
            criticality=0.5,
            answer_cost=0.2,
        )
        for field_id in ("parent_a", "parent_b", "target")
    }
    specs = [
        CompleteSpec(
            spec_id=f"weighted-{index}",
            values={
                "parent_a": _observed(0.0),
                "parent_b": _observed(float(index)),
                "target": _observed(target),
            },
        )
        for index, target in enumerate((0.0, 0.0, 0.0, 0.0, 10.0))
    ]
    config = V3RelationConfig(
        field_schema_path="unused.yaml",
        confidence_threshold=0.95,
        continuous_quantiles=(0.1, 0.5, 0.9),
        continuous_interval_quantiles=(0.025, 0.975),
        engineering_tolerances={"target": 0.1},
        relations=(
            RelationSpec(relation_id="empirical", family=RelationFamily.AFFINE, target_field_id="target", parent_field_ids=("parent_a",)),
            RelationSpec(relation_id="noisy", family=RelationFamily.AFFINE, target_field_id="target", parent_field_ids=("parent_b",)),
        ),
    )
    state = RequirementState(values={"parent_a": _observed(0.0), "parent_b": _observed(2.0), "target": _hidden()})

    posterior = TypedRelationalModel.fit(specs, schema, config).posterior(state, "target")

    assert posterior.branches[0].probability > 0.5
    assert posterior.branches[1].probability < 0.25
    assert posterior.confidence == posterior.branches[0].probability
