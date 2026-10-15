from pathlib import Path

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.policies.lookahead import (
    CCLAOneStepPolicy,
    CCLATwoStepPolicy,
    ConditionalAnswerModel,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED)


def test_conditional_answer_model_uses_observed_public_module_field():
    """防止模型忽略已知枢纽，退化为无条件的字段边际分布。"""
    training_specs = [
        {
            "upstream_output_interface": _observed("belt"),
            "station_input_interface": _observed("belt"),
        },
        {
            "upstream_output_interface": _observed("roller"),
            "station_input_interface": _observed("roller"),
        },
    ]
    model = ConditionalAnswerModel.fit(
        training_specs=training_specs,
        module_by_field={
            "upstream_output_interface": "interface",
            "station_input_interface": "interface",
        },
    )
    state = RequirementState(
        values={
            "upstream_output_interface": _observed("belt"),
            "station_input_interface": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )

    answers = model.answers_for(state, "station_input_interface")

    assert [(answer.value, answer.probability) for answer in answers] == [
        ("belt", 1.0)
    ]


def test_conditional_answer_model_falls_back_to_marginal_when_pivot_unknown():
    """防止模型在没有公开条件时虚构确定性回答。"""
    training_specs = [
        {
            "upstream_output_interface": _observed("belt"),
            "station_input_interface": _observed("belt"),
        },
        {
            "upstream_output_interface": _observed("roller"),
            "station_input_interface": _observed("roller"),
        },
    ]
    model = ConditionalAnswerModel.fit(
        training_specs=training_specs,
        module_by_field={
            "upstream_output_interface": "interface",
            "station_input_interface": "interface",
        },
    )
    state = RequirementState(
        values={
            "upstream_output_interface": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "station_input_interface": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )

    answers = model.answers_for(state, "station_input_interface")

    assert [(answer.value, answer.probability) for answer in answers] == [
        ("belt", 0.5),
        ("roller", 0.5),
    ]


def test_two_step_policy_selects_low_criticality_pivot_with_conditional_gain():
    """防止两步策略退化为按字段关键度的单步贪婪排序。"""
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    training_specs = [
        {
            "workpiece_fragile": _observed(True),
            "positioning_accuracy_mm": _observed(0.2),
            "repeatability_mm": _observed(0.1),
            "workpiece_mass_kg": _observed(10.0),
        },
        {
            "workpiece_fragile": _observed(False),
            "positioning_accuracy_mm": _observed(2.0),
            "repeatability_mm": _observed(1.0),
            "workpiece_mass_kg": _observed(20.0),
        },
    ]
    model = ConditionalAnswerModel.fit(
        training_specs=training_specs,
        module_by_field={
            "workpiece_fragile": "positioning",
            "positioning_accuracy_mm": "positioning",
            "repeatability_mm": "positioning",
            "workpiece_mass_kg": "load",
        },
    )
    state = RequirementState(
        values={
            field_id: FieldValue(value=None, status=FieldStatus.HIDDEN)
            for field_id in (
                "workpiece_fragile",
                "positioning_accuracy_mm",
                "repeatability_mm",
                "workpiece_mass_kg",
            )
        }
    )
    context = PolicyContext(field_schema=schema, budget=8, step_index=0)

    one_step = CCLAOneStepPolicy(
        engine=engine, answer_model=model, lambda_cost=0.0, lambda_solver=0.0
    )
    two_step = CCLATwoStepPolicy(
        engine=engine, answer_model=model, lambda_cost=0.0, lambda_solver=0.0
    )

    assert one_step.select(state, engine.check(state), context).field_id == "workpiece_mass_kg"
    decision = two_step.select(state, engine.check(state), context)
    assert decision.field_id == "workpiece_fragile"
    assert decision.utility_terms["lookahead_gain"] > 0.0
    assert decision.utility_terms["counterfactual_calls"] > 0.0
