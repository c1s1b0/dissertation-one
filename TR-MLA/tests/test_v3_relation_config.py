from pathlib import Path
from copy import deepcopy

import pytest
import yaml

from paper1_active_req.relations.config import RelationFamily, load_v3_relation_config
from paper1_active_req.relations.posterior import FieldPosterior, OutcomeBranch


def test_v3_relation_config_has_only_frozen_supported_families():
    config = load_v3_relation_config(Path("configs/v3_relations.yaml"))

    assert config.confidence_threshold == 0.95
    assert config.continuous_quantiles == (0.1, 0.5, 0.9)
    assert getattr(config, "continuous_interval_quantiles", None) == (0.025, 0.975)
    assert {item.family for item in config.relations} <= {
        RelationFamily.DISCRETE_CPT,
        RelationFamily.AFFINE,
        RelationFamily.RATIO,
        RelationFamily.INVERSE_TIME,
        RelationFamily.GATED_AFFINE,
    }
    assert all("audit" not in item.model_fields for item in config.relations)


def test_v3_relation_config_freezes_required_relation_mappings():
    config = load_v3_relation_config(Path("configs/v3_relations.yaml"))
    relations = {
        item.relation_id: (
            item.family,
            item.target_field_id,
            item.parent_field_ids,
            item.gate_field_id,
        )
        for item in config.relations
    }

    assert relations == {
        "upstream_to_station_input_interface": (
            RelationFamily.DISCRETE_CPT,
            "station_input_interface",
            ("upstream_output_interface",),
            None,
        ),
        "station_input_to_station_output_interface": (
            RelationFamily.DISCRETE_CPT,
            "station_output_interface",
            ("station_input_interface",),
            None,
        ),
        "station_to_downstream_input_interface": (
            RelationFamily.DISCRETE_CPT,
            "downstream_input_interface",
            ("station_output_interface",),
            None,
        ),
        "length_clearance": (
            RelationFamily.AFFINE,
            "available_length_mm",
            ("workpiece_length_mm",),
            None,
        ),
        "width_clearance": (
            RelationFamily.AFFINE,
            "available_width_mm",
            ("workpiece_width_mm",),
            None,
        ),
        "height_clearance": (
            RelationFamily.AFFINE,
            "available_height_mm",
            ("workpiece_height_mm",),
            None,
        ),
        "travel_speed_time": (
            RelationFamily.INVERSE_TIME,
            "transfer_time_s",
            ("travel_distance_mm", "conveyor_speed_mm_s"),
            None,
        ),
        "fragile_positioning": (
            RelationFamily.GATED_AFFINE,
            "positioning_accuracy_mm",
            ("repeatability_mm",),
            "workpiece_fragile",
        ),
    }


def test_v3_experiment_config_freezes_protocol_and_formal_paths():
    payload = yaml.safe_load(
        Path("configs/v3_experiment.yaml").read_text(encoding="utf-8")
    )

    assert payload["master_seed"] == 20260806
    assert payload["split"] == {"train": 180, "validation": 60, "test": 60}
    assert payload["missing_rates"] == [0.3, 0.5, 0.7]
    assert payload["budget"] == 8
    assert payload["lambda_candidates"] == [0.0, 0.05, 0.1, 0.2]
    assert payload["paths"]["generated_data"] == "data/generated/v3"
    assert payload["paths"]["tuning_results"] == "results/v3/tuning"
    assert payload["paths"]["formal_results"] == "results/v3/formal"


def _valid_relation_payload() -> dict:
    return {
        "field_schema_path": str(Path("configs/fields.yaml").resolve()),
        "confidence_threshold": 0.95,
        "continuous_quantiles": [0.1, 0.5, 0.9],
        "continuous_interval_quantiles": [0.025, 0.975],
        "engineering_tolerances": {"available_length_mm": 10.0},
        "relations": [
            {
                "relation_id": "length_clearance",
                "family": "affine",
                "target_field_id": "available_length_mm",
                "parent_field_ids": ["workpiece_length_mm"],
            }
        ],
    }


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "reversed continuous interval quantiles",
            lambda payload: payload.update(
                continuous_interval_quantiles=[0.975, 0.025]
            ),
        ),
        (
            "continuous interval quantiles outside unit interval",
            lambda payload: payload.update(
                continuous_interval_quantiles=[0.0, 0.975]
            ),
        ),
        (
            "duplicate relation identifier",
            lambda payload: payload["relations"].append(
                deepcopy(payload["relations"][0])
            ),
        ),
        (
            "unknown referenced field",
            lambda payload: payload["relations"][0].update(
                parent_field_ids=["not_a_schema_field"]
            ),
        ),
        (
            "target repeated as parent",
            lambda payload: payload["relations"][0].update(
                parent_field_ids=["available_length_mm"]
            ),
        ),
        (
            "continuous target without tolerance",
            lambda payload: payload.update(engineering_tolerances={}),
        ),
        (
            "unknown tolerance field",
            lambda payload: payload.update(
                engineering_tolerances={"not_a_schema_field": 1.0}
            ),
        ),
        (
            "non-continuous tolerance field",
            lambda payload: payload.update(
                engineering_tolerances={"station_input_interface": 1.0}
            ),
        ),
        (
            "not-a-number tolerance",
            lambda payload: payload.update(
                engineering_tolerances={"available_length_mm": float("nan")}
            ),
        ),
        (
            "positive infinity tolerance",
            lambda payload: payload.update(
                engineering_tolerances={"available_length_mm": float("inf")}
            ),
        ),
        (
            "negative infinity tolerance",
            lambda payload: payload.update(
                engineering_tolerances={"available_length_mm": float("-inf")}
            ),
        ),
        (
            "zero tolerance",
            lambda payload: payload.update(
                engineering_tolerances={"available_length_mm": 0.0}
            ),
        ),
        (
            "negative tolerance",
            lambda payload: payload.update(
                engineering_tolerances={"available_length_mm": -0.1}
            ),
        ),
    ],
)
def test_v3_relation_config_rejects_invalid_relation_contracts(
    tmp_path: Path, name: str, mutate
):
    """防止关系配置在训练前静默接受无效字段或连续变量精度。"""
    payload = _valid_relation_payload()
    mutate(payload)
    config_path = tmp_path / "relations.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_v3_relation_config(config_path)


def test_field_posterior_rejects_probability_mass_outside_tolerance():
    """防止后验分支没有构成合法概率分布却参与推理。"""
    with pytest.raises(ValueError, match="sum to 1"):
        FieldPosterior(
            field_id="station_input_interface",
            branches=(
                OutcomeBranch(value="belt", probability=0.6),
                OutcomeBranch(value="roller", probability=0.3),
            ),
            confidence=0.95,
        )


def test_field_posterior_accepts_probability_mass_within_tolerance():
    posterior = FieldPosterior(
        field_id="station_input_interface",
        branches=(
            OutcomeBranch(value="belt", probability=0.5),
            OutcomeBranch(value="roller", probability=0.5),
        ),
        confidence=0.95,
        factor_ids=("upstream_to_station_input_interface",),
    )

    assert posterior.factor_ids == ("upstream_to_station_input_interface",)


@pytest.mark.parametrize(
    ("interval_low", "interval_high"),
    [
        (2.0, 1.0),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (float("inf"), 1.0),
        (1.0, float("-inf")),
    ],
)
def test_field_posterior_rejects_invalid_interval_endpoints(
    interval_low: float, interval_high: float
):
    """防止无效区间端点传播到连续字段的后验计算。"""
    with pytest.raises(ValueError):
        FieldPosterior(
            field_id="available_length_mm",
            branches=(OutcomeBranch(value=1000.0, probability=1.0),),
            confidence=0.95,
            interval_low=interval_low,
            interval_high=interval_high,
        )
