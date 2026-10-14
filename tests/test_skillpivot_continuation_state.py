"""Exercise continuation state through Pipeline, not only identity helpers."""

import json
from dataclasses import dataclass, field
from functools import partial

import pytest

from skillopt.envs.base import EnvAdapter
from skillopt.optimizer.skillpivot import (
    ContinuationEnvironment,
    DeltaGenerator,
    RegressionGatedVerifier,
    SkillPivotPipeline,
    TeacherContinuationEngine,
    teacher_continuation,
)
from skillopt.types import PairedSuffixCase


@dataclass
class Door:
    opened: bool = False
    actions: list = field(default_factory=list)

    def step(self, action):
        self.actions.append(action)
        if action == "open":
            self.opened = True
            return "door opened", False, 0.0
        if action == "complete" or (action == "collect" and self.opened):
            return "item collected", True, 1.0
        return "door closed", True, 0.0


class DefaultAdapter(EnvAdapter):
    def build_train_env(self, *args, **kwargs):
        return []

    def build_eval_env(self, *args, **kwargs):
        return []

    def rollout(self, *args, **kwargs):
        return []

    def get_task_types(self):
        return ["interactive"]


@pytest.fixture
def pipeline_factory(monkeypatch):
    def make(actions, retries=3):
        responses = iter(actions)
        model_calls = []

        def chat(**kwargs):
            model_calls.append(kwargs)
            return f"<action>{next(responses)}</action>", {}

        def delta_chat(stage="", **kwargs):
            if stage == "contrast_analysis":
                data = {
                    "student_mistake": "Stopped prematurely",
                    "teacher_correction": "Completed the task",
                    "skill_gap": "Missing result check",
                    "evolution_direction": "Verify completion",
                }
            else:
                data = {
                    "action": "improve_skill",
                    "content": "Verify the task result before stopping.",
                    "trigger_condition": "When ending a task",
                }
            return json.dumps(data), {}

        monkeypatch.setattr(teacher_continuation, "chat_optimizer", chat)
        pipeline = SkillPivotPipeline(
            teacher_engine=TeacherContinuationEngine(max_steps=1, max_retries=retries),
            delta_generator=DeltaGenerator(chat_fn=delta_chat),
            verifier=RegressionGatedVerifier(
                chat_fn=lambda **kwargs: (json.dumps({"no_regression_score": 1, "generalizable_rule_score": 1}), {})
            ),
        )
        return pipeline, model_calls

    return make


def run(pipeline, **kwargs):
    case = PairedSuffixCase(task_id="door")
    skill, deltas, gates = pipeline.optimize_step([case], "# Skill", **kwargs)
    return case, skill, deltas, gates


@pytest.mark.parametrize("wrapper", ["bound_closure", "partial", "default_argument", "decorator"])
@pytest.mark.parametrize("retries", [2, 3, 4])
def test_wrappers_cannot_hide_dirty_state(pipeline_factory, wrapper, retries):
    door = Door()

    def build(attempt):
        bound_step = door.step
        if wrapper == "partial":
            return partial(bound_step)
        if wrapper == "default_argument":
            return lambda action, step=bound_step: step(action)
        if wrapper == "decorator":

            def decorated(action):
                return bound_step(action)

            decorated.__wrapped__ = bound_step
            return decorated
        return lambda action: bound_step(action)

    pipeline, calls = pipeline_factory(["open", "collect"], retries=retries)
    case, skill, deltas, gates = run(pipeline, env_step_fn_builder=build)
    assert not case.teacher_success
    assert door.actions == ["open"]
    assert len(calls) == 1
    assert skill == "# Skill" and not deltas and not gates
    assert [step.action for step in case.teacher_suffix] == ["open"]


@pytest.mark.parametrize("recorder_kind", ["list", "bound_helper", "function"])
def test_clean_environment_may_share_audit_dependencies(pipeline_factory, recorder_kind):
    trace = []
    configuration = {"audit": True}
    record = trace.append
    if recorder_kind == "function":

        def record(entry):
            trace.append(entry)

    environments = []

    def build(attempt):
        door = Door()
        environments.append(door)

        def step(action):
            if configuration["audit"]:
                if recorder_kind == "list":
                    trace.append((attempt, action))
                else:
                    record((attempt, action))
            return door.step(action)

        return step

    pipeline, calls = pipeline_factory(["wrong", "complete"])
    case, skill, deltas, gates = run(pipeline, env_step_fn_builder=build)
    assert case.teacher_success and len(calls) == 2
    assert trace == [(1, "wrong"), (2, "complete")]
    assert len(environments) == 2 and environments[0] is not environments[1]
    assert len(deltas) == len(gates) == 1 and skill != "# Skill"


def test_explicit_handle_separates_state_from_adapter_and_logs(pipeline_factory):
    class Adapter(DefaultAdapter):
        def __init__(self):
            self.trace = []

        def build_continuation_step_fn(self, case, **kwargs):
            door = Door()

            def step(action):
                self.trace.append(action)
                return door.step(action)

            return ContinuationEnvironment(step, state=door)

    adapter = Adapter()
    pipeline, _ = pipeline_factory(["wrong", "complete"])
    case, _, deltas, _ = run(pipeline, adapter=adapter)
    assert case.teacher_success and len(deltas) == 1
    assert adapter.trace == ["wrong", "complete"]


def test_consumed_restore_certificate_cannot_authorize_another_attempt(pipeline_factory):
    door = Door()
    handle = ContinuationEnvironment(door.step, state=door, restored=True)
    pipeline, calls = pipeline_factory(["open", "collect"])
    case, _, deltas, _ = run(pipeline, env_step_fn_builder=lambda attempt: handle)
    assert not case.teacher_success and not deltas
    assert len(calls) == 1 and door.actions == ["open"]


def test_opaque_shared_dictionary_requires_restore_proof(pipeline_factory):
    state = {"opened": False}

    def build(attempt):
        def step(action):
            if action == "open":
                state["opened"] = True
                return "opened", False, 0.0
            return "collected", True, float(state["opened"])

        return step

    pipeline, calls = pipeline_factory(["open", "collect"])
    case, _, deltas, _ = run(pipeline, env_step_fn_builder=build)
    assert not case.teacher_success and not deltas and len(calls) == 1


@pytest.mark.parametrize("reset_failure", ["exception", "false"])
def test_explicit_restoration_recovers_same_opaque_state(pipeline_factory, reset_failure):
    state = {"opened": False}

    class Adapter(DefaultAdapter):
        def reset_continuation_env(self, case, **kwargs):
            if reset_failure == "exception":
                raise RuntimeError("old checkpoint unavailable")
            return False

    def build(attempt):
        state["opened"] = False  # restore the saved prefix for this build

        def step(action):
            if action == "open":
                state["opened"] = True
                return "opened", False, 0.0
            return "completed", True, float(action == "complete")

        return ContinuationEnvironment(step, state=state, restored=True)

    pipeline, calls = pipeline_factory(["open", "complete"])
    case, _, deltas, _ = run(pipeline, env_step_fn_builder=build, adapter=Adapter())
    assert case.teacher_success and len(deltas) == 1 and len(calls) == 2


def test_reset_business_typeerror_is_not_replayed(pipeline_factory):
    class BrokenResetDoor(Door):
        def __init__(self):
            super().__init__()
            self.resets = []

        def reset(self, attempt=1):
            self.resets.append(attempt)
            raise TypeError("reset implementation failed")

    door = BrokenResetDoor()
    pipeline, calls = pipeline_factory(["open", "collect"])
    case, _, deltas, _ = run(pipeline, env_step_fn=door.step)
    assert not case.teacher_success and not deltas
    assert door.resets == [2, 3] and len(calls) == 1


def test_hook_business_typeerror_is_not_treated_as_signature_mismatch(pipeline_factory):
    class BrokenAdapter(DefaultAdapter):
        def __init__(self):
            self.calls = []

        def build_continuation_step_fn(self, case, train_env=None, attempt=1):
            self.calls.append(attempt)
            raise TypeError("checkpoint parsing failed")

    adapter = BrokenAdapter()
    pipeline, calls = pipeline_factory([])
    case, _, deltas, _ = run(pipeline, adapter=adapter)
    assert not case.teacher_success and not deltas and not calls
    assert adapter.calls == [1, 2, 3]
    assert case.teacher_suffix[0].action == "ENVIRONMENT_ERROR"


def test_successful_reset_cleans_only_its_own_environment(pipeline_factory):
    class ResettableDoor(Door):
        def __init__(self):
            super().__init__()
            self.resets = []

        def reset(self, attempt):
            self.resets.append(attempt)
            self.opened = False

    doors = [Door(), ResettableDoor()]

    def build(attempt):
        return doors[0].step if attempt in (1, 4) else doors[1].step

    pipeline, calls = pipeline_factory(["open", "wrong", "wrong", "collect"], retries=4)
    case, _, deltas, _ = run(pipeline, env_step_fn_builder=build)
    assert not case.teacher_success and not deltas
    assert len(calls) == 3 and doors[0].actions == ["open"]
    assert doors[1].actions == ["wrong", "wrong"] and doors[1].resets == [3]


def test_pipeline_forwards_explicit_reset_callback_once(pipeline_factory):
    door = Door()
    resets = []

    def reset(*, attempt):
        resets.append(attempt)
        door.opened = False

    pipeline, _ = pipeline_factory(["open", "complete"])
    case, _, deltas, _ = run(pipeline, env_step_fn=door.step, env_reset_fn=reset)
    assert case.teacher_success and len(deltas) == 1
    assert resets == [2] and door.actions == ["open", "complete"]


def test_environment_cannot_downgrade_to_qa_after_a_failed_attempt(pipeline_factory):
    door = Door()
    pipeline, calls = pipeline_factory(["open", "FINISH: 4"])
    case = PairedSuffixCase(task_id="downgrade", gold_answers=["4"])
    _, deltas, _ = pipeline.optimize_step(
        [case],
        "# Skill",
        env_step_fn_builder=lambda attempt: door.step if attempt == 1 else None,
    )
    assert not case.teacher_success and not deltas and len(calls) == 1


def test_explicit_evaluator_is_not_overridden_by_adapter(pipeline_factory):
    class Adapter(DefaultAdapter):
        def build_continuation_evaluator_fn(self, case, **kwargs):
            raise AssertionError("explicit evaluator must take priority")

    pipeline, _ = pipeline_factory(["FINISH: 4"] * 3)
    case = PairedSuffixCase(task_id="evaluator", gold_answers=["4"])
    _, deltas, _ = pipeline.optimize_step(
        [case],
        "# Skill",
        adapter=Adapter(),
        evaluator_fn=lambda answer, gold: False,
    )
    assert not case.teacher_success and not deltas


@pytest.mark.parametrize("failure", ["exception", "invalid_result"])
def test_failed_adapter_evaluator_never_downgrades_to_answer_matching(pipeline_factory, failure):
    class Adapter(DefaultAdapter):
        def build_continuation_evaluator_fn(self, case, **kwargs):
            if failure == "exception":
                raise TypeError("evaluator construction failed")
            return "not a callable"

    pipeline, calls = pipeline_factory([])
    case = PairedSuffixCase(task_id="evaluator-error", gold_answers=["4"])
    _, deltas, gates = pipeline.optimize_step([case], "# Skill", adapter=Adapter())
    assert not case.teacher_success and not deltas and not gates and not calls
    assert "continuation_evaluator_error" in case.extras


def test_trainer_resolves_adapter_evaluator_with_case_and_real_context(tmp_path, pipeline_factory):
    from skillopt.engine.trainer import run_skillpivot_reflect_step
    from tests.test_deviation_detector import MockEmbeddingProvider

    received = []
    train_env = {"checkpoint": "training-batch"}

    class Adapter(DefaultAdapter):
        def build_continuation_evaluator_fn(self, case, train_env=None):
            assert case is not None
            received.append((case.task_id, train_env))
            return lambda answer, gold: False

    task_dir = tmp_path / "failure"
    task_dir.mkdir()
    (task_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"step": 1, "action": "wrong", "env_feedback": "failed", "execution_status": 0},
            ]
        ),
        encoding="utf-8",
    )
    pipeline, calls = pipeline_factory(["FINISH: 4"] * 3)
    patches = run_skillpivot_reflect_step(
        [{"id": "failure", "hard": 0, "gold_answers": ["4"]}],
        "# Skill",
        str(tmp_path),
        {},
        adapter=Adapter(),
        train_env=train_env,
        embedding_provider=MockEmbeddingProvider(),
        pipeline=pipeline,
    )
    assert patches == [] and len(calls) == 3
    assert len(received) == 1 and received[0][0] == "failure"
    assert received[0][1] is train_env
