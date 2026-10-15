from __future__ import annotations

from paper1_active_req.domain.models import (
    FieldDefinition,
    FieldKind,
    FieldStatus,
    FieldValue,
    RequirementState,
)
from paper1_active_req.generator.v3_relational_specs import (
    V3AuditMetadata,
    V3RequirementTask,
)
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.solver.engine import SolverResult


class PublicEpisodeEngine:
    def __init__(self, schema):
        self.schema = schema

    def check(self, state):
        del state
        return SolverResult(satisfiable=True)


class GuardedV3Task:
    """任何读取 audit 或 ground_truth 的 episode 回归都会立即失败。"""

    task_id = "v3-guarded-episode"

    def __init__(self) -> None:
        self.initial_state = RequirementState(
            values={
                "question": FieldValue(
                    value=None, status=FieldStatus.HIDDEN, source="guarded-task"
                )
            }
        )

    @property
    def audit(self):
        raise AssertionError("episode must not read task.audit")

    @property
    def ground_truth(self):
        raise AssertionError("episode must not read task.ground_truth")


class PublicV3Policy:
    name = "public_v3_probe"

    def select(self, state, solver_result, context):
        from paper1_active_req.experiment.v3_episode import V3PolicyDecision

        del solver_result
        assert state.values["question"].value is None
        assert not hasattr(context, "audit")
        assert not hasattr(context, "ground_truth")
        return V3PolicyDecision(
            field_id="question",
            utility_terms={"marginal_score": 0.5, "first_step_gain": 0.5},
            relation_factor_ids=("clearance_affine",),
            resolved_before=(),
            marginal_score=0.5,
        )


class MisreportingV3Policy(PublicV3Policy):
    name = "misreporting_v3_probe"

    def select(self, state, solver_result, context):
        decision = super().select(state, solver_result, context)
        return decision.model_copy(update={"resolved_before": ("forged",)})


def run_v3_episode_fixture():
    from paper1_active_req.experiment.v3_episode import run_v3_episode

    definition = FieldDefinition(
        field_id="question",
        kind=FieldKind.CATEGORICAL,
        categories=("low", "high"),
        criticality=1.0,
        answer_cost=1.0,
    )
    task = V3RequirementTask(
        task_id="v3-public-episode",
        spec_id="v3-public-spec",
        split="validation",
        missing_rate=0.5,
        initial_state=RequirementState(
            values={
                "question": FieldValue(
                    value=None, status=FieldStatus.HIDDEN, source="fixture"
                )
            }
        ),
        ground_truth={
            "question": FieldValue(
                value="low", status=FieldStatus.OBSERVED, source="fixture"
            )
        },
        audit=V3AuditMetadata(
            module_id="v3.continuous_clearance",
            pivot_field_id="question",
            affected_critical_field_ids=(),
            mechanism="continuous_clearance",
        ),
    )
    return run_v3_episode(
        task=task,
        policy=PublicV3Policy(),
        oracle=SimulatedOracle(task.ground_truth),
        engine=PublicEpisodeEngine({"question": definition}),
        budget=1,
    )


def test_v3_episode_preserves_relation_factor_ids_without_ground_truth():
    episode = run_v3_episode_fixture()

    assert episode.steps[0].relation_factor_ids == ("clearance_affine",)
    assert episode.steps[0].resolved_before == ()
    assert episode.steps[0].resolved_after == ("question",)
    assert episode.steps[0].marginal_score == 0.5
    assert "ground_truth" not in episode.steps[0].model_dump()
    assert "ground_truth" not in episode.model_dump_json()


def test_v3_episode_uses_guarded_task_and_computes_resolved_before_itself():
    from paper1_active_req.experiment.v3_episode import run_v3_episode

    definition = FieldDefinition(
        field_id="question",
        kind=FieldKind.CATEGORICAL,
        categories=("low", "high"),
        criticality=1.0,
        answer_cost=1.0,
    )
    episode = run_v3_episode(
        task=GuardedV3Task(),
        policy=MisreportingV3Policy(),
        oracle=SimulatedOracle(
            {
                "question": FieldValue(
                    value="low", status=FieldStatus.OBSERVED, source="oracle-fixture"
                )
            }
        ),
        engine=PublicEpisodeEngine({"question": definition}),
        budget=1,
    )

    assert episode.steps[0].resolved_before == ()
    assert episode.steps[0].answer.value == "low"
