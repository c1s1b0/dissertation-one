from pathlib import Path

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import PolicyContext
from paper1_active_req.policies.base import (
    EmpiricalAnswerModel,
    WeightedAnswer,
)
from paper1_active_req.policies.baselines import (
    ConstraintRangeOnlyPolicy,
    CriticalityGreedyPolicy,
    FixedQuestionnairePolicy,
    RandomPolicy,
    UncertaintyAFAPolicy,
)
from paper1_active_req.solver.engine import ConstraintEngine


def _context_and_state():
    schema = load_field_schema(Path("configs/fields.yaml"))
    state = RequirementState(
        values={
            "workpiece_mass_kg": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "positioning_accuracy_mm": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "side_clearance_mm": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
            "power_supply": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )
    context = PolicyContext(
        field_schema=schema,
        budget=8,
        step_index=0,
    )
    return schema, state, context


def test_criticality_policy_selects_highest_criticality():
    schema, state, context = _context_and_state()
    result = ConstraintEngine(schema).check(state)

    decision = CriticalityGreedyPolicy().select(state, result, context)

    assert decision.field_id == "positioning_accuracy_mm"
    assert decision.utility_terms == {"criticality": 1.0}


def test_random_policy_is_reproducible_and_skips_nonqueryable_field():
    schema, state, context = _context_and_state()
    result = ConstraintEngine(schema).check(state)

    first = RandomPolicy(seed=7).select(state, result, context)
    second = RandomPolicy(seed=7).select(state, result, context)

    assert first.field_id == second.field_id
    assert first.field_id != "power_supply"


def test_fixed_questionnaire_uses_first_available_configured_field():
    schema, state, context = _context_and_state()
    result = ConstraintEngine(schema).check(state)
    policy = FixedQuestionnairePolicy(
        order=["side_clearance_mm", "workpiece_mass_kg"]
    )

    decision = policy.select(state, result, context)

    assert decision.field_id == "side_clearance_mm"


def test_uncertainty_policy_uses_empirical_training_distribution():
    schema, state, context = _context_and_state()
    result = ConstraintEngine(schema).check(state)
    model = EmpiricalAnswerModel(
        outcomes={
            "workpiece_mass_kg": [
                WeightedAnswer(value=20.0, probability=1.0)
            ],
            "positioning_accuracy_mm": [
                WeightedAnswer(value=0.2, probability=0.5),
                WeightedAnswer(value=1.0, probability=0.5),
            ],
            "side_clearance_mm": [
                WeightedAnswer(value=50.0, probability=0.8),
                WeightedAnswer(value=100.0, probability=0.2),
            ],
        }
    )

    decision = UncertaintyAFAPolicy(model).select(
        state, result, context
    )

    assert decision.field_id == "positioning_accuracy_mm"
    assert set(decision.utility_terms) == {"entropy"}


def test_constraint_range_policy_returns_auditable_gain():
    schema, state, context = _context_and_state()
    engine = ConstraintEngine(schema)
    result = engine.check(state)
    model = EmpiricalAnswerModel(
        outcomes={
            field_id: [WeightedAnswer(value=value, probability=1.0)]
            for field_id, value in {
                "workpiece_mass_kg": 20.0,
                "positioning_accuracy_mm": 0.5,
                "side_clearance_mm": 80.0,
            }.items()
        }
    )

    decision = ConstraintRangeOnlyPolicy(engine, model).select(
        state, result, context
    )

    assert decision.field_id in {
        "workpiece_mass_kg",
        "positioning_accuracy_mm",
        "side_clearance_mm",
    }
    assert set(decision.utility_terms) == {"domain_reduction"}
