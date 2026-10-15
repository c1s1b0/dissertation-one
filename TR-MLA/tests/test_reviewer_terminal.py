import pytest
from test_v3_metrics import _task_and_schema, _episode, SatisfiableEngine
from paper1_active_req.policies.base import PolicyDecision


def test_final_query_replacement_does_not_mutate_original_or_reveal_answer():
    from paper1_active_req.experiment.reviewer_terminal import finish_episode
    task, schema = _task_and_schema()
    original = _episode(task)
    before = original.model_dump_json()
    class LastPolicy:
        def select(self, state, solver_result, context):
            assert state.values['pivot'].value is None
            assert context.budget == 1 and context.step_index == 0
            return PolicyDecision(field_id='pivot')
    engine = SatisfiableEngine()
    engine.schema = schema
    result = finish_episode(original, task, LastPolicy(), engine, budget=1)
    assert result.steps[0].field_id == 'pivot'
    assert result.steps[0].answer.value == 'left'
    assert result.final_state.values['critical'].value is None
    assert original.model_dump_json() == before


def test_short_episode_is_retained_and_wrong_task_rejected():
    from paper1_active_req.experiment.reviewer_terminal import finish_episode
    task, schema = _task_and_schema()
    original = _episode(task)
    class NeverCalled:
        def select(self, *args):
            raise AssertionError('no extra query after original early stop')
    assert finish_episode(original, task, NeverCalled(), SatisfiableEngine(), budget=8).steps == original.steps
    task.task_id = 'wrong'
    with pytest.raises(ValueError, match='task'):
        finish_episode(original, task, NeverCalled(), SatisfiableEngine(), budget=8)
