from pathlib import Path

import numpy as np

from paper1_active_req.domain.models import FieldStatus
from paper1_active_req.domain.schema import load_field_schema
from paper1_active_req.experiment.episode import run_episode
from paper1_active_req.generator.complete_specs import generate_complete_spec
from paper1_active_req.generator.perturbations import (
    inject_conflict,
    mask_spec,
)
from paper1_active_req.oracle.simulated import SimulatedOracle
from paper1_active_req.policies.base import PolicyDecision
from paper1_active_req.solver.engine import ConstraintEngine


class CheatingPolicy:
    name = "cheating_probe"

    def select(self, state, solver_result, context):
        assert not hasattr(context, "ground_truth")
        assert all(
            item.value is None
            for item in state.values.values()
            if item.status == FieldStatus.HIDDEN
        )
        return None


class FirstQueryablePolicy:
    name = "first_queryable"

    def select(self, state, solver_result, context):
        del solver_result, context
        candidates = state.queryable_field_ids()
        return PolicyDecision(
            field_id=candidates[0] if candidates else None,
            utility_terms={"fixed": 1.0},
        )


def _build_task(seed: int = 21):
    schema = load_field_schema(Path("configs/fields.yaml"))
    engine = ConstraintEngine(schema)
    rng = np.random.default_rng(seed)
    spec = generate_complete_spec(rng, schema, engine, "episode-spec")
    task = mask_spec(spec, missing_rate=0.5, rng=rng)
    return schema, engine, task


def test_policy_context_never_contains_ground_truth():
    _, engine, task = _build_task()
    oracle = SimulatedOracle(task.ground_truth)

    result = run_episode(
        task=task,
        policy=CheatingPolicy(),
        oracle=oracle,
        engine=engine,
        budget=8,
    )

    assert result.steps == []
    assert result.solver_calls == 1


def test_oracle_answer_replaces_conflicting_value():
    _, engine, task = _build_task(22)
    rng = np.random.default_rng(23)
    task = inject_conflict(task, "input_interface", rng)
    oracle = SimulatedOracle(task.ground_truth)
    field_id = task.injected_conflict_field

    answer = oracle.answer(field_id)

    assert answer.status == FieldStatus.OBSERVED
    assert answer.value == task.ground_truth[field_id].value
    assert answer.source == "oracle"
    assert engine.check(task.initial_state).satisfiable is False


def test_episode_applies_answer_without_mutating_task():
    _, engine, task = _build_task(24)
    original = task.initial_state.model_copy(deep=True)
    oracle = SimulatedOracle(task.ground_truth)

    result = run_episode(
        task=task,
        policy=FirstQueryablePolicy(),
        oracle=oracle,
        engine=engine,
        budget=1,
    )

    assert len(result.steps) == 1
    asked = result.steps[0].field_id
    assert result.final_state.values[asked].status == FieldStatus.OBSERVED
    assert task.initial_state == original
    assert result.solver_calls == 2
