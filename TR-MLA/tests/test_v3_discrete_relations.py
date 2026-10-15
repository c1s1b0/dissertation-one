from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.generator.complete_specs import CompleteSpec
from paper1_active_req.relations.config import RelationFamily, RelationSpec
from paper1_active_req.relations import discrete
from paper1_active_req.relations.discrete import DiscreteConditionalFactor


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED, source="fixture")


def _hidden() -> FieldValue:
    return FieldValue(value=None, status=FieldStatus.HIDDEN, source="fixture")


def interface_relation() -> RelationSpec:
    return RelationSpec(
        relation_id="interface_chain",
        family=RelationFamily.DISCRETE_CPT,
        target_field_id="station_output_interface",
        parent_field_ids=("station_input_interface", "power_supply"),
    )


def training_interface_specs() -> list[CompleteSpec]:
    values = (
        [("roller", "220vac", "roller")] * 40
        + [("roller", "380vac", "roller")] * 5
        + [("belt", "220vac", "belt")] * 5
        + [("belt", "380vac", "belt")] * 5
        + [("pallet", "220vac", "pallet")] * 5
        + [("pallet", "24vdc", "pallet")] * 5
    )
    return [
        CompleteSpec(
            spec_id=f"training-{index:03d}",
            values={
                "station_input_interface": _observed(interface),
                "power_supply": _observed(power),
                "station_output_interface": _observed(output),
            },
        )
        for index, (interface, power, output) in enumerate(values)
    ]


def state_with(interface: str | None, power: str | None) -> RequirementState:
    return RequirementState(
        values={
            "station_input_interface": _observed(interface)
            if interface is not None
            else _hidden(),
            "power_supply": _observed(power) if power is not None else _hidden(),
        }
    )


def _specs(
    rows: list[tuple[str, str, str]],
) -> list[CompleteSpec]:
    return [
        CompleteSpec(
            spec_id=f"fixture-{index:03d}",
            values={
                "station_input_interface": _observed(interface),
                "power_supply": _observed(power),
                "station_output_interface": _observed(output),
            },
        )
        for index, (interface, power, output) in enumerate(rows)
    ]


def _relation(
    parent_field_ids: tuple[str, ...], minimum_support: int = 3
) -> RelationSpec:
    return RelationSpec(
        relation_id="interface_chain",
        family=RelationFamily.DISCRETE_CPT,
        target_field_id="station_output_interface",
        parent_field_ids=parent_field_ids,
        minimum_support=minimum_support,
    )


def _distribution(posterior) -> tuple[tuple[object, ...], tuple[float, ...]]:
    return (
        tuple(branch.value for branch in posterior.branches),
        tuple(branch.probability for branch in posterior.branches),
    )


def test_discrete_factor_uses_joint_parents_then_information_ranked_fallback():
    factor = DiscreteConditionalFactor.fit(training_interface_specs(), interface_relation())

    joint = factor.posterior(state_with(interface="roller", power="220vac"))
    fallback = factor.posterior(state_with(interface="roller", power=None))

    assert joint.branches[0].value == "roller"
    assert joint.confidence >= 0.95
    assert factor.fallback_parent_field_ids == ("station_input_interface", "power_supply")
    assert fallback.factor_ids == ("interface_chain",)
    assert sum(branch.probability for branch in fallback.branches) == pytest.approx(1.0)


def test_discrete_factor_falls_back_for_unseen_joint_condition_without_empty_branches():
    factor = DiscreteConditionalFactor.fit(training_interface_specs(), interface_relation())

    posterior = factor.posterior(state_with(interface="roller", power="24vdc"))

    assert posterior.branches
    assert posterior.branches[0].value == "roller"
    assert sum(branch.probability for branch in posterior.branches) == pytest.approx(1.0)


def test_discrete_factor_rejects_training_category_outside_schema():
    specs = training_interface_specs()
    specs[0] = specs[0].model_copy(
        update={
            "values": {
                **specs[0].values,
                "station_output_interface": _observed("not-an-interface"),
            }
        }
    )

    with pytest.raises(ValueError, match="schema"):
        DiscreteConditionalFactor.fit(specs, interface_relation())


def test_discrete_factor_uses_higher_mi_parent_despite_reverse_declaration_order():
    rows = (
        [("roller", "220vac", "roller")] * 4
        + [("roller", "380vac", "roller")] * 4
        + [("belt", "220vac", "belt")] * 4
        + [("belt", "380vac", "belt")] * 4
        + [("pallet", "24vdc", "pallet")] * 8
    )
    factor = DiscreteConditionalFactor.fit(
        _specs(rows),
        _relation(("power_supply", "station_input_interface")),
    )

    posterior = factor.posterior(state_with(interface="roller", power="24vdc"))

    assert factor.fallback_parent_field_ids == (
        "station_input_interface",
        "power_supply",
    )
    assert _distribution(posterior) == (
        ("roller", "belt", "pallet"),
        pytest.approx((9 / 11, 1 / 11, 1 / 11)),
    )


def test_discrete_factor_breaks_exact_mi_ties_by_field_id():
    rows = [
        *( [("roller", "220vac", "roller")] * 4),
        *( [("belt", "380vac", "belt")] * 4),
    ]
    factor = DiscreteConditionalFactor.fit(
        _specs(rows),
        _relation(("station_input_interface", "power_supply")),
    )

    posterior = factor.posterior(state_with(interface="belt", power="220vac"))

    assert factor.fallback_parent_field_ids == (
        "power_supply",
        "station_input_interface",
    )
    assert _distribution(posterior) == (
        ("roller", "belt", "pallet"),
        pytest.approx((5 / 7, 1 / 7, 1 / 7)),
    )


def test_discrete_factor_selects_joint_single_and_marginal_at_support_threshold():
    rows = (
        [("roller", "220vac", "roller")] * 2
        + [("roller", "380vac", "roller")] * 2
        + [("belt", "220vac", "belt")] * 4
        + [("pallet", "24vdc", "pallet")] * 3
    )
    factor = DiscreteConditionalFactor.fit(
        _specs(rows),
        _relation(("station_input_interface", "power_supply"), minimum_support=4),
    )

    below_joint = factor.posterior(state_with(interface="roller", power="220vac"))
    exact_joint = factor.posterior(state_with(interface="belt", power="220vac"))
    below_single = factor.posterior(state_with(interface=None, power="24vdc"))

    assert _distribution(below_joint) == (
        ("roller", "belt", "pallet"),
        pytest.approx((5 / 7, 1 / 7, 1 / 7)),
    )
    assert _distribution(exact_joint) == (
        ("belt", "roller", "pallet"),
        pytest.approx((5 / 7, 1 / 7, 1 / 7)),
    )
    assert _distribution(below_single) == (
        ("belt", "roller", "pallet"),
        pytest.approx((5 / 14, 5 / 14, 4 / 14)),
    )


def test_discrete_factor_smooths_schema_category_missing_from_training():
    factor = DiscreteConditionalFactor.fit(
        _specs([("roller", "220vac", "roller")] * 2),
        _relation(("station_input_interface", "power_supply"), minimum_support=1),
    )

    posterior = factor.posterior(state_with(interface="roller", power="220vac"))

    values, probabilities = _distribution(posterior)
    assert values == ("roller", "belt", "pallet")
    assert probabilities == pytest.approx((3 / 5, 1 / 5, 1 / 5))
    assert len(values) == len(set(values)) == 3
    assert all(probability > 0.0 for probability in probabilities)


def test_discrete_factor_rejects_duplicate_parent_field_ids():
    with pytest.raises(ValueError, match="duplicate parent"):
        DiscreteConditionalFactor.fit(
            training_interface_specs(),
            _relation(("station_input_interface", "station_input_interface")),
        )


def test_discrete_factor_uses_packaged_schema_and_snapshots_domains(monkeypatch):
    monkeypatch.setattr(
        discrete,
        "_SCHEMA_PATH",
        Path("schema-is-not-available-from-installed-wheel.yaml"),
        raising=False,
    )
    factor = DiscreteConditionalFactor.fit(training_interface_specs(), interface_relation())
    monkeypatch.setattr(
        discrete,
        "load_field_schema",
        lambda _: pytest.fail("posterior must use the fitted schema snapshot"),
        raising=False,
    )

    posterior = factor.posterior(state_with(interface="roller", power=None))

    assert posterior.branches[0].value == "roller"


def test_packaged_discrete_schema_domains_match_public_field_schema():
    resource = files("paper1_active_req.relations").joinpath(
        "resources/v3_discrete_schema.yaml"
    )
    packaged = yaml.safe_load(resource.read_text(encoding="utf-8"))["fields"]
    public_schema = load_field_schema(Path("configs/fields.yaml"))

    assert {
        (item["field_id"], item["kind"], tuple(item.get("categories", ())))
        for item in packaged
    } == {
        (field_id, definition.kind.value, definition.categories)
        for field_id, definition in public_schema.items()
        if definition.kind.value in {"categorical", "boolean"}
    }
