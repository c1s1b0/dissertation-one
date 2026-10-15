from pathlib import Path

import pytest

from paper1_active_req.domain.models import (
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.metrics import (
    critical_requirement_recall,
    episode_metrics,
    recall_budget_auc,
)
from paper1_active_req.evaluation.statistics import holm_adjust
from paper1_active_req.experiment.episode import EpisodeResult, EpisodeStep
from paper1_active_req.generator.complete_specs import RequirementTask
from paper1_active_req.solver.engine import ConstraintEngine


def _task_and_result():
    hidden = {
        field_id: FieldValue(value=None, status=FieldStatus.HIDDEN)
        for field_id in [
            "workpiece_mass_kg",
            "positioning_accuracy_mm",
            "side_clearance_mm",
        ]
    }
    ground_truth = {
        "workpiece_mass_kg": FieldValue(
            value=20.0,
            status=FieldStatus.OBSERVED,
            source="ground_truth",
        ),
        "positioning_accuracy_mm": FieldValue(
            value=0.5,
            status=FieldStatus.OBSERVED,
            source="ground_truth",
        ),
        "side_clearance_mm": FieldValue(
            value=80.0,
            status=FieldStatus.OBSERVED,
            source="ground_truth",
        ),
    }
    initial = RequirementState(values=hidden)
    after_mass = initial.apply_answer(
        "workpiece_mass_kg",
        ground_truth["workpiece_mass_kg"].model_copy(
            update={"source": "oracle"}
        ),
    )
    final = after_mass.apply_answer(
        "positioning_accuracy_mm",
        ground_truth["positioning_accuracy_mm"].model_copy(
            update={"source": "oracle"}
        ),
    )
    steps = [
        EpisodeStep(
            step_index=0,
            field_id="workpiece_mass_kg",
            utility_terms={},
            answer=final.values["workpiece_mass_kg"],
            satisfiable_after_answer=True,
        ),
        EpisodeStep(
            step_index=1,
            field_id="positioning_accuracy_mm",
            utility_terms={},
            answer=final.values["positioning_accuracy_mm"],
            satisfiable_after_answer=True,
        ),
    ]
    task = RequirementTask(
        task_id="metric-task",
        spec_id="metric-spec",
        split="test",
        missing_rate=0.5,
        initial_state=initial,
        ground_truth=ground_truth,
    )
    result = EpisodeResult(
        task_id=task.task_id,
        policy_name="test",
        initial_state=initial,
        final_state=final,
        steps=steps,
        solver_calls=4,
    )
    return task, result


def test_weighted_crr_uses_initially_unresolved_critical_fields():
    schema = load_field_schema(Path("configs/fields.yaml"))
    task, result = _task_and_result()

    at_one = critical_requirement_recall(
        result, task, schema, budget=1
    )
    at_eight = critical_requirement_recall(
        result, task, schema, budget=8
    )

    assert at_one == pytest.approx(0.5)
    assert at_eight == pytest.approx(1.0)


def test_recall_budget_auc_is_normalized():
    schema = load_field_schema(Path("configs/fields.yaml"))
    task, result = _task_and_result()

    score = recall_budget_auc(
        result, task, schema, max_budget=8
    )

    assert 0.0 <= score <= 1.0
    assert score > 0.8


def test_episode_metrics_report_constraint_and_cost_state():
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    task, result = _task_and_result()

    metrics = episode_metrics(result, task, schema, engine)

    assert set(metrics) >= {
        "crr_at_8",
        "recall_budget_auc",
        "hard_constraint_satisfied",
        "conflict_detected",
        "unsupported_assumption_rate",
        "answer_cost",
        "solver_calls",
    }
    assert metrics["answer_cost"] == pytest.approx(0.6)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
