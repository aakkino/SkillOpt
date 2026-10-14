"""Tests for skillopt.optimizer.skillpivot modules."""
import json
import pytest

from skillopt.optimizer.skillpivot import (
    DeltaGenerator,
    RegressionGatedVerifier,
    SkillPivotPipeline,
    TeacherContinuationEngine,
)
from skillopt.types import (
    DeviationPoint,
    PairedSuffixCase,
    RegressionGateResult,
    SkillDelta,
    TrajectoryStep,
)


def make_sample_case() -> PairedSuffixCase:
    """Helper to create a populated sample PairedSuffixCase."""
    prefix = [
        TrajectoryStep(step=1, action="search A", observation="obs A", reasoning="think 1"),
        TrajectoryStep(step=2, action="search B", observation="obs B", reasoning="think 2"),
    ]
    student_sfx = [
        TrajectoryStep(step=3, action="repeat click", observation="error 404", execution_status=0),
        TrajectoryStep(step=4, action="repeat click", observation="error 404", execution_status=0),
    ]
    dev = DeviationPoint(
        step_index=3,
        trough_potential=0.1,
        pre_mean_potential=0.8,
        post_mean_potential=0.1,
        is_confirmed=True,
    )
    return PairedSuffixCase(
        task_id="case_001",
        task_description="Search for quarterly revenue",
        ground_truth="revenue is $5M",
        skill_name="search_skill",
        skill_content="# Base Search Skill\nAlways use precise keywords.",
        deviation=dev,
        prefix_steps=prefix,
        student_suffix=student_sfx,
        teacher_suffix=[],
        teacher_success=False,
    )


def test_teacher_continuation_prompt_building():
    """Verify prompt formatting includes shared prefix and negative reference."""
    engine = TeacherContinuationEngine()
    case = make_sample_case()

    prompt = engine.build_teacher_prefix_prompt(case)
    assert "## Task Description" in prompt
    assert "[Step 1 Action] search A" in prompt
    assert "## Negative Reference (What to Avoid)" in prompt
    assert "repeat click" in prompt
    assert "starting from Step 3" in prompt


def test_teacher_continuation_simulated_execution():
    """Verify teacher continuation runs and finishes successfully."""
    engine = TeacherContinuationEngine(max_steps=5, allow_simulated=True)
    case = make_sample_case()

    # Mock chat_fn returning finish on step 2
    turn_count = 0

    def mock_teacher_chat(system, user, **kwargs):
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            return "<think>Let me query the balance sheet</think><action>query balance</action>", {}
        return "<think>Found revenue</think><action>FINISH: $5M</action>", {}

    res_case = engine.run_continuation(case, chat_fn=mock_teacher_chat)

    assert res_case.teacher_success is True
    assert len(res_case.teacher_suffix) == 2
    assert res_case.teacher_suffix[0].step == 3
    assert res_case.teacher_suffix[1].step == 4
    assert res_case.teacher_suffix[0].action == "query balance"
    assert res_case.teacher_suffix[1].action == "FINISH: $5M"


def test_teacher_continuation_verification_and_rejection():
    """Verify teacher rejection on wrong answers, zero-reward stops, and unverifiable steps (P1 fix)."""
    engine = TeacherContinuationEngine(max_steps=3, allow_simulated=False)

    # 1. Wrong answer rejection with ground_truth
    case_math = PairedSuffixCase(
        task_id="case_math",
        task_description="What is 2+2?",
        ground_truth="4",
    )
    def mock_wrong_chat(system=None, user=None, **kw):
        return "<action>FINISH: 5</action>", {}

    res_wrong = engine.run_continuation(case_math, chat_fn=mock_wrong_chat)
    assert res_wrong.teacher_success is False

    # 2. Correct answer acceptance with ground_truth
    def mock_correct_chat(system=None, user=None, **kw):
        return "<action>FINISH: 4</action>", {}

    res_correct = engine.run_continuation(case_math, chat_fn=mock_correct_chat)
    assert res_correct.teacher_success is True

    # 3. Environment terminated with zero reward (e.g. wrong answer, done=True, reward=0.0)
    def mock_env_zero_reward(action):
        return "wrong answer", True, 0.0

    res_env_fail = engine.run_continuation(
        case_math,
        env_step_fn=mock_env_zero_reward,
        chat_fn=lambda system=None, user=None, **kw: ("<action>submit 5</action>", {}),
    )
    assert res_env_fail.teacher_success is False

    # 4. Environment terminated with positive reward (reward=1.0)
    def mock_env_success(action):
        return "success", True, 1.0

    res_env_ok = engine.run_continuation(
        case_math,
        env_step_fn=mock_env_success,
        chat_fn=lambda system=None, user=None, **kw: ("<action>submit 4</action>", {}),
    )
    assert res_env_ok.teacher_success is True

    # 5. Production mode without env_step_fn or ground_truth must reject continuation
    case_no_gt = PairedSuffixCase(
        task_id="case_no_gt",
        task_description="Open ended exploration",
        ground_truth="",
    )
    res_no_verif = engine.run_continuation(
        case_no_gt,
        chat_fn=lambda system=None, user=None, **kw: ("<action>FINISH: done</action>", {}),
        allow_simulated=False,
    )
    assert res_no_verif.teacher_success is False



def test_delta_generator_contrast_and_apply():
    """Verify delta generation and application to skill document."""
    # Mock chat_fn returning contrast summary and skill delta
    call_idx = 0

    def mock_delta_chat(system, user, stage="", **kwargs):
        nonlocal call_idx
        call_idx += 1
        if stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "repeated click without checking 404",
                "teacher_correction": "used query alternative",
                "skill_gap": "missing fallback for 404 response",
                "evolution_direction": "retry with alternative keyword",
            }), {}
        return json.dumps({
            "action": "improve_skill",
            "target_rule": "Error Handling",
            "content": "When encountering a 404 response, immediately switch to broader synonyms.",
            "trigger_condition": "404 tool error",
            "rationale": "Prevents loop failure",
        }), {}

    generator = DeltaGenerator(chat_fn=mock_delta_chat)
    case = make_sample_case()
    case.teacher_suffix = [
        TrajectoryStep(step=3, action="query alternative", observation="success", execution_status=1),
    ]
    case.teacher_success = True

    delta = generator.generate_delta(case)
    assert delta.action == "improve_skill"
    assert "404" in delta.content

    # Test applying delta
    orig_skill = "# Base Skill\nStep 1."
    tentative = generator.apply_delta(orig_skill, delta)
    assert "# Base Skill" in tentative
    assert "When encountering a 404 response" in tentative
    assert "Applicable Condition: 404 tool error" in tentative


def test_regression_verifier_gate_logic():
    """Verify acceptance gate logic when scores >= 0.5."""
    # Test Pass
    def mock_chat_pass(system, user, **kwargs):
        return json.dumps({
            "no_regression_score": 0.9,
            "generalizable_rule_score": 0.8,
            "critique": "Preserves rules and adds reusable recovery.",
        }), {}

    verifier = RegressionGatedVerifier(threshold=0.5, chat_fn=mock_chat_pass)
    gate_pass = verifier.verify_update(
        original_skill="Base skill",
        tentative_skill="Base skill + robust rule",
    )
    assert gate_pass.is_accepted is True
    assert gate_pass.no_regression_score == 0.9

    # Test Reject (regression score too low)
    def mock_chat_fail(system, user, **kwargs):
        return json.dumps({
            "no_regression_score": 0.3,
            "generalizable_rule_score": 0.7,
            "critique": "Deleted crucial existing constraints.",
        }), {}

    verifier_fail = RegressionGatedVerifier(threshold=0.5, chat_fn=mock_chat_fail)
    gate_fail = verifier_fail.verify_update(
        original_skill="Base skill with constraints",
        tentative_skill="Deleted constraints",
    )
    assert gate_fail.is_accepted is False


def test_skillpivot_end_to_end_pipeline():
    """Verify complete SkillPivotPipeline execution."""
    pipeline = SkillPivotPipeline()

    case = make_sample_case()

    # Pre-populate teacher suffix
    case.teacher_suffix = [
        TrajectoryStep(step=3, action="query alternative", observation="success", execution_status=1),
    ]
    case.teacher_success = True

    # Mock generator and verifier inside pipeline
    pipeline.delta_generator.chat_fn = lambda system, user, stage="", **kw: (
        json.dumps({
            "action": "improve_skill",
            "target_rule": "Fallback",
            "content": "Verify intermediate results before final stop.",
            "trigger_condition": "Empty tool response",
            "rationale": "Stops premature exit",
        }),
        {},
    )
    pipeline.verifier.chat_fn = lambda system, user, **kw: (
        json.dumps({
            "no_regression_score": 0.85,
            "generalizable_rule_score": 0.85,
            "critique": "Solid reusable guideline.",
        }),
        {},
    )

    base_skill = "# Base Skill\nStep 1: start."
    new_skill, deltas, gate_results = pipeline.optimize_step([case], base_skill)

    assert len(deltas) == 1
    assert len(gate_results) == 1
    assert gate_results[0].is_accepted is True
    assert "Verify intermediate results" in new_skill
    assert "# Base Skill\nStep 1: start." in new_skill
