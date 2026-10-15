from pathlib import Path

from paper1_active_req.domain.models import FieldStatus, FieldValue, RequirementState
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.evaluation.v2_metrics import v2_critical_requirement_recall
from paper1_active_req.experiment.episode import EpisodeResult
from paper1_active_req.generator.v2_pivotal_specs import (
    V2AuditMetadata,
    V2RequirementTask,
)
from paper1_active_req.policies.lookahead import ConditionalAnswerModel


def _observed(value: object) -> FieldValue:
    return FieldValue(value=value, status=FieldStatus.OBSERVED)


def _task() -> V2RequirementTask:
    initial_state = RequirementState(
        values={
            "workpiece_fragile": _observed(True),
            "positioning_accuracy_mm": FieldValue(
                value=None, status=FieldStatus.HIDDEN
            ),
        }
    )
    return V2RequirementTask(
        task_id="metric-task",
        spec_id="v2-spec-metric",
        split="test",
        missing_rate=0.5,
        initial_state=initial_state,
        ground_truth={
            "workpiece_fragile": _observed(True),
            "positioning_accuracy_mm": _observed(0.2),
        },
        audit=V2AuditMetadata(
            module_id="positioning",
            pivot_field_id="workpiece_fragile",
            affected_critical_field_ids=("positioning_accuracy_mm",),
            mechanism="conditional_correlation",
        ),
    )


def _result(task: V2RequirementTask) -> EpisodeResult:
    return EpisodeResult(
        task_id=task.task_id,
        policy_name="ccla_two_step",
        initial_state=task.initial_state,
        final_state=task.initial_state,
        steps=[],
        solver_calls=0,
    )


def test_v2_resolved_crr_counts_correct_unique_inference_but_not_wrong_one():
    """防止指标将模型的唯一猜测当作真值恢复。"""
    task = _task()
    schema = load_field_schema(Path("configs/fields.yaml"))
    correct_model = ConditionalAnswerModel.fit(
        training_specs=[
            {
                "workpiece_fragile": _observed(True),
                "positioning_accuracy_mm": _observed(0.2),
            }
        ],
        module_by_field={
            "workpiece_fragile": "positioning",
            "positioning_accuracy_mm": "positioning",
        },
    )
    wrong_model = ConditionalAnswerModel.fit(
        training_specs=[
            {
                "workpiece_fragile": _observed(True),
                "positioning_accuracy_mm": _observed(0.3),
            }
        ],
        module_by_field={
            "workpiece_fragile": "positioning",
            "positioning_accuracy_mm": "positioning",
        },
    )

    assert v2_critical_requirement_recall(
        _result(task), task, schema, correct_model, budget=0
    ) == 1.0
    assert v2_critical_requirement_recall(
        _result(task), task, schema, wrong_model, budget=0
    ) == 0.0
