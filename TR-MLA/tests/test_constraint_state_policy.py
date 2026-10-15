from pathlib import Path

import numpy as np
import pytest

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.generator.complete_specs import generate_complete_spec
from paper1_active_req.generator.perturbations import (
    inject_conflict,
    mask_spec,
)
from paper1_active_req.policies.base import EmpiricalAnswerModel
from paper1_active_req.policies.constraint_state import (
    ConstraintStatePolicy,
    UtilityWeights,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _single_value_model(ground_truth):
    return EmpiricalAnswerModel.from_complete_specs(
        values=[ground_truth],
        max_numeric_outcomes=5,
    )


def test_full_policy_prioritizes_field_in_unsat_core():
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(31)
    spec = generate_complete_spec(rng, schema, engine, "conflict-spec")
    task = mask_spec(spec, missing_rate=0.5, rng=rng)
    task = inject_conflict(task, "input_interface", rng)
    model = _single_value_model(task.ground_truth)
    context = PolicyContext(
        field_schema=schema,
        budget=8,
        step_index=0,
    )
    policy = ConstraintStatePolicy(
        engine=engine,
        empirical_model=model,
        weights=UtilityWeights(
            domain_reduction=1.0,
            criticality=1.0,
            conflict=4.0,
            graph_degree=0.5,
            answer_cost=0.5,
        ),
    )

    result = engine.check(task.initial_state)
    decision = policy.select(task.initial_state, result, context)

    assert decision.field_id == "station_input_interface"
    assert decision.utility_terms["conflict"] == 1.0
    assert set(decision.utility_terms) == {
        "domain_reduction",
        "criticality",
        "conflict",
        "graph_degree",
        "answer_cost",
        "total",
    }


def test_answer_cost_breaks_equal_gain_tie():
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    state = RequirementState(
        values={
            "workpiece_mass_kg": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "side_clearance_mm": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )
    model = EmpiricalAnswerModel.from_complete_specs(
        values=[
            {
                "workpiece_mass_kg": FieldValue(
                    value=20.0, status=FieldStatus.OBSERVED
                ),
                "side_clearance_mm": FieldValue(
                    value=80.0, status=FieldStatus.OBSERVED
                ),
            }
        ]
    )
    context = PolicyContext(
        field_schema=schema,
        budget=8,
        step_index=0,
    )
    policy = ConstraintStatePolicy(
        engine=engine,
        empirical_model=model,
        weights=UtilityWeights(
            domain_reduction=0.0,
            criticality=0.0,
            conflict=0.0,
            graph_degree=0.0,
            answer_cost=1.0,
        ),
    )

    decision = policy.select(state, engine.check(state), context)

    assert decision.field_id == "workpiece_mass_kg"
    assert decision.utility_terms["answer_cost"] == pytest.approx(0.0)


def test_empirical_model_normalizes_probabilities():
    values = [
        {
            "workpiece_mass_kg": FieldValue(
                value=value, status=FieldStatus.OBSERVED
            ),
            "station_input_interface": FieldValue(
                value=interface, status=FieldStatus.OBSERVED
            ),
        }
        for value, interface in [
            (10.0, "belt"),
            (20.0, "belt"),
            (30.0, "roller"),
        ]
    ]

    model = EmpiricalAnswerModel.from_complete_specs(
        values=values,
        max_numeric_outcomes=2,
    )

    assert len(model.answers_for("workpiece_mass_kg")) == 2
    assert sum(
        item.probability
        for item in model.answers_for("workpiece_mass_kg")
    ) == pytest.approx(1.0)
    assert {
        item.value for item in model.answers_for("station_input_interface")
    } == {"belt", "roller"}
