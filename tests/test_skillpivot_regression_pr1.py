"""Comprehensive regression tests covering PR #1 review findings:
1. Strict teacher verification rejecting partial numbers (40 vs 4), empty XML tags, and punctuation.
2. Full multi-gold answers verification (NY vs New York).
3. Student context (target_user_prompt.txt) preservation in teacher continuation prompt without gold leakage.
4. Downstream patch application consistency with protected regions (APPENDIX).
5. Post-aggregation & selection candidate semantic gate review and rejection.
"""
from __future__ import annotations

import json
import os
import pytest

from skillopt.config import flatten_config, load_config
from skillopt.deviation_detector import DeviationDetector
from skillopt.engine.trainer import run_skillpivot_reflect_step
from skillopt.optimizer.skill import apply_edit, apply_patch_with_report
from skillopt.optimizer.skillpivot import (
    DeltaGenerator,
    RegressionGatedVerifier,
    SkillPivotPipeline,
    TeacherContinuationEngine,
)
from skillopt.optimizer.skillpivot.teacher_continuation import (
    default_verify_answer,
    extract_final_answer,
)
from skillopt.types import (
    DeviationPoint,
    PairedSuffixCase,
    RegressionGateResult,
    SkillDelta,
    TrajectoryStep,
)
from tests.test_deviation_detector import MockEmbeddingProvider


# ── 1. Strict Teacher Verification & Empty Rejection ─────────────────────────


def test_strict_teacher_verification_and_empty_rejection():
    """Verify strict exact match rejecting partial numbers, empty tags, and punctuation."""
    # 1. Partial number must be rejected
    assert default_verify_answer("40", "4") is False
    assert default_verify_answer("40", ["4"]) is False
    assert default_verify_answer("14", "4") is False
    assert default_verify_answer("4", "40") is False

    # 2. Exact match must pass
    assert default_verify_answer("4", "4") is True
    assert default_verify_answer("4", ["4"]) is True
    assert default_verify_answer("4", ["4", "four"]) is True

    # 3. Empty and punctuation-only predictions must be rejected
    assert default_verify_answer("", "4") is False
    assert default_verify_answer("   ", "4") is False
    assert default_verify_answer("...", "4") is False
    assert default_verify_answer("!?", "4") is False

    # 4. Extraction helpers
    assert extract_final_answer("<answer>40</answer>") == "40"
    assert extract_final_answer("<answer></answer>") == ""
    assert extract_final_answer("<answer>   </answer>") == ""
    assert extract_final_answer("<action>FINISH: 40</action>") == "40"
    assert extract_final_answer("<action>FINISH: </action>") == ""
    assert extract_final_answer("<action>FINISH:   </action>") == ""

    # 5. SearchQA evaluator behaviour via run_skillpivot_reflect_step default evaluator
    from skillopt.envs.searchqa.evaluator import evaluate as searchqa_eval

    # Substring in searchqa_eval might give sub_em=1.0 for empty or partial, but runner strictly checks em >= 1.0 and non-empty
    sub_eval_40 = searchqa_eval("40", ["4"])
    assert sub_eval_40["em"] == 0.0

    sub_eval_empty = searchqa_eval("<answer></answer>", ["4"])
    assert sub_eval_empty["em"] == 0.0


# ── 2. Full Multi-Gold Answers Verification ──────────────────────────────────


def test_multi_gold_answers_verification():
    """Verify teacher succeeds when answering secondary gold aliases."""
    engine = TeacherContinuationEngine(max_steps=3, allow_simulated=False)

    case = PairedSuffixCase(
        task_id="multi_gold_task",
        task_description="What is the capital of New York State?",
        ground_truth="Albany",
        gold_answers=["Albany, NY", "City of Albany", "Albany"],
        deviation=DeviationPoint(step_index=1, trough_potential=0.1, pre_mean_potential=0.0, post_mean_potential=0.1),
    )

    # Teacher answers a secondary alias: "City of Albany"
    def mock_chat_alias(system, user, **kw):
        return "<think>Found city</think><action>FINISH: City of Albany</action>", {}

    res_case = engine.run_continuation(case, chat_fn=mock_chat_alias)
    assert res_case.teacher_success is True

    # Teacher answers third alias: "Albany"
    def mock_chat_short(system, user, **kw):
        return "<think>Found city</think><action>FINISH: Albany</action>", {}

    res_case_2 = engine.run_continuation(case, chat_fn=mock_chat_short)
    assert res_case_2.teacher_success is True

    # Teacher answers wrong city: "Buffalo"
    def mock_chat_wrong(system, user, **kw):
        return "<think>Wrong guess</think><action>FINISH: Buffalo</action>", {}

    res_case_wrong = engine.run_continuation(case, chat_fn=mock_chat_wrong)
    assert res_case_wrong.teacher_success is False


# ── 3. Student Context Preservation & No Gold Leakage ─────────────────────────


def test_student_context_preservation_and_teacher_prompt(tmp_path):
    """Verify target_user_prompt.txt is recovered and fed into teacher continuation prompt without leaking gold."""
    pred_dir = tmp_path / "predictions"
    task_dir = pred_dir / "task_searchqa_context"
    task_dir.mkdir(parents=True)

    # Write target_user_prompt.txt containing evidence context
    user_prompt_content = (
        "## Context\n"
        "PRIVATE_FACT_982: The required figure is 4.\n\n"
        "## Question\n"
        "What is the required figure?"
    )
    with open(task_dir / "target_user_prompt.txt", "w", encoding="utf-8") as f:
        f.write(user_prompt_content)

    # Conversation with failure
    conversation = [
        {"type": "message", "turn": 1, "content": "<think>Guessing</think><answer>5</answer>"},
        {"role": "system", "content": "[EVALUATION RESULT]\nExact Match: 0.0"},
    ]
    with open(task_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    rollout_results = [
        {
            "id": "task_searchqa_context",
            "question": "What is the required figure?",
            "gold_answers": ["4"],
            "hard": 0.0,
            "soft": 0.0,
            "response": "<answer>5</answer>",
        }
    ]

    teacher_prompt_captured = []

    def mock_chat(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            teacher_prompt_captured.append(user)
            return "<think>Reading context</think><action>FINISH: 4</action>", {}
        if stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Guessed 5",
                "teacher_correction": "Extracted 4 from context",
                "skill_gap": "Need to check PRIVATE_FACT",
                "evolution_direction": "Ground answers in context",
            }), {}
        if stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Context Grounding",
                "content": "Extract numbers strictly from context.",
                "trigger_condition": "Numerical question",
                "rationale": "Avoid hallucination",
            }), {}
        if stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Solid guideline",
            }), {}
        return "", {}

    cfg = {
        "optimizer_type": "skillpivot",
        "skillpivot_embedding_backend": "local",
        "skillpivot_gate_threshold": 0.5,
    }

    from skillopt.optimizer.skillpivot import teacher_continuation, delta_generator, regression_verifier

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat)

    try:
        raw_patches = run_skillpivot_reflect_step(
            rollout_results=rollout_results,
            current_skill="# QA Skill\nAnswer questions directly.",
            prediction_dir=str(pred_dir),
            cfg=cfg,
            embedding_provider=MockEmbeddingProvider(),
        )

        assert len(raw_patches) == 1
        assert len(teacher_prompt_captured) > 0
        prompt = teacher_prompt_captured[0]

        # Assert context is preserved
        assert "PRIVATE_FACT_982: The required figure is 4." in prompt

        # Assert gold evaluation was not leaked into teacher prompt
        assert "[EVALUATION RESULT]" not in prompt
        assert "Exact Match: 0.0" not in prompt
    finally:
        monkeypatch.undo()


# ── 4. Protected Region & apply_delta Consistency ─────────────────────────────


def test_protected_region_and_apply_delta_consistency():
    """Verify apply_delta uses downstream apply_edit logic, placing appends before APPENDIX."""
    skill_with_appendix = (
        "# Core Skill Guide\n\n"
        "Rule 1: Always check inputs.\n\n"
        "<!-- APPENDIX_START -->\n"
        "## APPENDIX: Protected Constants\n"
        "CONSTANT_TIMEOUT = 30\n"
        "<!-- APPENDIX_END -->\n"
    )

    delta = SkillDelta(
        action="improve_skill",
        skill_name="active_skill",
        target_rule="Retry Protocol",
        content="Retry operation when idempotent.",
        trigger_condition="Only if idempotent",
        rationale="Avoid duplicate side-effects",
    )

    # Tentative skill generated by DeltaGenerator
    tentative_skill = DeltaGenerator.apply_delta(skill_with_appendix, delta)

    # Text applied via downstream apply_patch_with_report
    patch_dict = {
        "edits": [
            {
                "op": "append",
                "content": delta.render_content(),
            }
        ]
    }
    downstream_skill, report = apply_patch_with_report(skill_with_appendix, patch_dict)

    # 1. Byte-level consistency between pipeline review candidate and trainer downstream applied skill
    assert tentative_skill == downstream_skill

    # 2. Both must place the procedural update BEFORE <!-- APPENDIX_START -->
    assert "### Procedural Update [improve_skill]: Retry Protocol" in tentative_skill
    assert "> Applicable Condition: Only if idempotent" in tentative_skill

    appendix_pos = tentative_skill.find("<!-- APPENDIX_START -->")
    update_pos = tentative_skill.find("### Procedural Update")
    assert update_pos != -1 and appendix_pos != -1
    assert update_pos < appendix_pos, "The append edit must be inserted before the protected region"


# ── 5. Post-Aggregation Candidate Semantic Gate Rejection ─────────────────────


def test_final_candidate_semantic_gate_rejection():
    """Verify that when merged/rewritten patches violate regression gate, the verifier rejects them."""
    current_skill = "# Operations Skill\n1. Check status before proceeding."

    # A harmful candidate that removes conditions or introduces regression
    harmful_candidate = "# Operations Skill\n1. Retry every operation infinitely without check."

    verifier = RegressionGatedVerifier(threshold=0.5)

    def mock_reject_chat(system, user, **kw):
        return json.dumps({
            "no_regression_score": 0.2,
            "generalizable_rule_score": 0.3,
            "critique": "Deletes status check and introduces infinite retry regressions.",
        }), {}

    verifier.chat_fn = mock_reject_chat

    res = verifier.verify_update(
        original_skill=current_skill,
        tentative_skill=harmful_candidate,
    )

    assert res.is_accepted is False
    assert res.no_regression_score == 0.2
    assert res.generalizable_rule_score == 0.3
    assert "infinite retry regressions" in res.critique
    # Verify to_dict output
    d = res.to_dict()
    assert d["is_accepted"] is False
    assert d["no_regression_score"] == 0.2
    assert d["generalizable_rule_score"] == 0.3


# ── 6. RegressionGateResult Serialization Round-Trip ─────────────────────────


def test_regression_gate_result_serialization():
    """Verify RegressionGateResult to_dict and from_dict round-trip serialization."""
    res_accept = RegressionGateResult(
        no_regression_score=0.88,
        generalizable_rule_score=0.92,
        threshold=0.5,
        critique="Good general rule.",
    )
    assert res_accept.is_accepted is True
    d_accept = res_accept.to_dict()
    assert d_accept == {
        "no_regression_score": 0.88,
        "generalizable_rule_score": 0.92,
        "threshold": 0.5,
        "is_accepted": True,
        "critique": "Good general rule.",
    }

    # Deserialization
    reconstructed = RegressionGateResult.from_dict(d_accept)
    assert reconstructed.is_accepted is True
    assert reconstructed.no_regression_score == 0.88
    assert reconstructed.generalizable_rule_score == 0.92
    assert reconstructed.threshold == 0.5
    assert reconstructed.critique == "Good general rule."

    # Rejection case
    res_reject = RegressionGateResult(
        no_regression_score=0.4,
        generalizable_rule_score=0.9,
        threshold=0.5,
        critique="Potential regression.",
    )
    assert res_reject.is_accepted is False
    d_reject = res_reject.to_dict()
    assert d_reject["is_accepted"] is False
    assert RegressionGateResult.from_dict(d_reject).is_accepted is False


# ── 7. Trainer SkillPivot Final Candidate Gate Lifecycle ──────────────────────


def test_trainer_skillpivot_final_candidate_gate_lifecycle(tmp_path, monkeypatch):
    """Verify ReflACTTrainer invokes final candidate semantic gate, serializes to_dict() without error,
    and correctly handles both accepted and rejected branches."""
    from skillopt.engine.trainer import ReflACTTrainer
    import skillopt.engine.trainer as trainer_mod

    # Setup temporary skill and output directory
    skill_file = tmp_path / "init_skill.md"
    skill_file.write_text("# Initial Skill\n1. Check inputs.")

    cfg_base = {
        "out_root": str(tmp_path / "run_out"),
        "skill_init": str(skill_file),
        "batch_size": 1,
        "num_epochs": 1,
        "accumulation": 1,
        "seed": 42,
        "merge_batch_size": 2,
        "train_size": 1,
        "edit_budget": 2,
        "eval_test": False,
        "use_gate": True,
        "gate_metric": "hard",
        "gate_mixed_weight": 0.5,
        "optimizer_type": "skillpivot",
        "skillpivot_gate_threshold": 0.5,
        "backend": "openai_compatible",
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": "mock-model",
        "target_model": "mock-model",
        "openai_compatible_base_url": "https://api.mock.com/v1",
        "openai_compatible_api_key": "mock-key",
        "analyst_workers": 1,
        "sel_env_num": 1,
        "val_env_num": 1,
    }

    from skillopt.envs.base import EnvAdapter

    # Common adapter mock
    class MockAdapter(EnvAdapter):
        def build_train_env(self, batch_size: int, seed: int, **kwargs):
            return ["dummy_train_env"]

        def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
            return ["dummy_eval_env"], 1

        def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
            return [{"id": "task_01", "hard": 1.0, "soft": 1.0}]

        def get_task_types(self) -> list[str]:
            return ["general"]

        def compute_score(self, results):
            return 1.0, 1.0

    # Ensure run_skillpivot_reflect_step produces a valid edit so candidate_skill != current_skill
    def mock_reflect_step(*args, **kwargs):
        return [
            {
                "patch": {
                    "reasoning": "SkillPivot procedural enhancement",
                    "edits": [
                        {
                            "op": "append",
                            "content": "\n### Procedural Update [improve_skill]\nRule: Validate input before writing.",
                            "target": "Rule: Validate input before writing.",
                            "source_type": "failure",
                        }
                    ],
                },
                "source_type": "failure",
                "batch_size": 1,
            }
        ]

    monkeypatch.setattr(trainer_mod, "run_skillpivot_reflect_step", mock_reflect_step)

    # ── Branch 1: Accepted by final semantic gate ────────────────────────────
    cfg_accept = dict(cfg_base, out_root=str(tmp_path / "run_accept"))

    def mock_verify_accept(self, original_skill, tentative_skill, delta=None):
        return RegressionGateResult(
            no_regression_score=0.95,
            generalizable_rule_score=0.90,
            threshold=0.5,
            critique="Update is solid and generalizable.",
        )

    monkeypatch.setattr(
        "skillopt.optimizer.skillpivot.regression_verifier.RegressionGatedVerifier.verify_update",
        mock_verify_accept,
    )

    trainer_accept = ReflACTTrainer(cfg_accept, MockAdapter())
    trainer_accept.train()

    # Verify step 1 produced step_record.json with skillpivot_final_gate_result dict
    step1_dir = tmp_path / "run_accept" / "steps" / "step_0001"
    assert step1_dir.exists()

    step_rec_file = step1_dir / "step_record.json"
    assert step_rec_file.exists()
    with open(step_rec_file, "r", encoding="utf-8") as f:
        step_rec = json.load(f)

    assert "skillpivot_final_gate_result" in step_rec
    gate_res_accept = step_rec["skillpivot_final_gate_result"]
    assert gate_res_accept["is_accepted"] is True
    assert gate_res_accept["no_regression_score"] == 0.95
    assert gate_res_accept["generalizable_rule_score"] == 0.90
    assert gate_res_accept["critique"] == "Update is solid and generalizable."

    # Verify dedicated skillpivot_final_gate_result.json file
    gate_json_file = step1_dir / "skillpivot_final_gate_result.json"
    assert gate_json_file.exists()
    with open(gate_json_file, "r", encoding="utf-8") as f:
        gate_json = json.load(f)
    assert gate_json == gate_res_accept

    # ── Branch 2: Rejected by final semantic gate ────────────────────────────
    cfg_reject = dict(cfg_base, out_root=str(tmp_path / "run_reject"))

    def mock_verify_reject(self, original_skill, tentative_skill, delta=None):
        return RegressionGateResult(
            no_regression_score=0.15,
            generalizable_rule_score=0.20,
            threshold=0.5,
            critique="Causes catastrophic regression.",
        )

    monkeypatch.setattr(
        "skillopt.optimizer.skillpivot.regression_verifier.RegressionGatedVerifier.verify_update",
        mock_verify_reject,
    )

    trainer_reject = ReflACTTrainer(cfg_reject, MockAdapter())
    trainer_reject.train()

    step1_reject_dir = tmp_path / "run_reject" / "steps" / "step_0001"
    assert step1_reject_dir.exists()

    with open(step1_reject_dir / "step_record.json", "r", encoding="utf-8") as f:
        step_rec_rej = json.load(f)

    assert step_rec_rej["action"] == "reject_by_skillpivot_gate"
    assert "skillpivot_final_gate_result" in step_rec_rej
    gate_res_rej = step_rec_rej["skillpivot_final_gate_result"]
    assert gate_res_rej["is_accepted"] is False
    assert gate_res_rej["no_regression_score"] == 0.15
    assert gate_res_rej["critique"] == "Causes catastrophic regression."

    # Verify skill remained unchanged on rejection
    with open(tmp_path / "run_reject" / "skills" / "skill_v0001.md", "r", encoding="utf-8") as f:
        v1_skill = f.read()
    assert v1_skill.strip() == "# Initial Skill\n1. Check inputs.".strip()


# ── 8. Teacher Continuation Environment Step Done & Reward Semantics ──────────


def test_teacher_continuation_env_step_reward_and_done_semantics():
    """Verify teacher continuation accepts positive reward ONLY upon true task completion (done=True).

    Specifically tests:
    1. Intermediate shaping reward (reward > 0, done=False): teacher does NOT terminate early with 1 step.
    2. Terminal success (reward > 0, done=True): teacher successfully completes task with full trajectory.
    3. Zero-reward termination (reward = 0, done=True): teacher terminates as unsuccessful.
    4. Custom threshold (success_reward_threshold): sub-threshold terminal rewards are rejected.
    """
    engine = TeacherContinuationEngine(
        max_steps=5,
        max_retries=1,
        allow_simulated=False,
        success_reward_threshold=0.0,
    )

    base_case = PairedSuffixCase(
        task_id="env_step_task",
        task_description="Search for document and complete task.",
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    # 1. Intermediate shaping reward (reward > 0, done=False) must NOT trigger early success
    step_calls = []

    def mock_env_shaping_then_done(action):
        step_calls.append(action)
        if len(step_calls) == 1:
            # Step 1: positive shaping reward, but NOT done
            return "Shaping progress: found candidate", False, 0.2
        elif len(step_calls) == 2:
            # Step 2: intermediate progress, still NOT done
            return "Shaping progress: reading page", False, 0.3
        else:
            # Step 3: true terminal completion
            return "Task completed successfully", True, 1.0

    def mock_chat_search(system, user, **kw):
        return f"<think>Step {len(step_calls)+1} plan</think><action>read_step_{len(step_calls)+1}</action>", {}

    res_case = engine.run_continuation(
        base_case,
        env_step_fn=mock_env_shaping_then_done,
        chat_fn=mock_chat_search,
    )

    assert res_case.teacher_success is True
    # Crucial assertion: must NOT stop at step 1 despite reward=0.2 > 0.0
    assert len(res_case.teacher_suffix) == 3
    assert len(step_calls) == 3
    assert res_case.teacher_suffix[0].reward == 0.2
    assert res_case.teacher_suffix[0].done is False
    assert res_case.teacher_suffix[1].reward == 0.3
    assert res_case.teacher_suffix[1].done is False
    assert res_case.teacher_suffix[2].reward == 1.0
    assert res_case.teacher_suffix[2].done is True

    # 2. Terminal success (reward > 0, done=True)
    def mock_env_one_step_success(action):
        return "Target reached", True, 1.0

    res_case_success = engine.run_continuation(
        base_case,
        env_step_fn=mock_env_one_step_success,
        chat_fn=mock_chat_search,
    )
    assert res_case_success.teacher_success is True
    assert len(res_case_success.teacher_suffix) == 1
    assert res_case_success.teacher_suffix[0].done is True
    assert res_case_success.teacher_suffix[0].reward == 1.0

    # 3. Zero-reward termination (reward = 0, done=True)
    def mock_env_zero_reward_done(action):
        return "Timeout: step limit reached without reaching goal", True, 0.0

    res_case_fail = engine.run_continuation(
        base_case,
        env_step_fn=mock_env_zero_reward_done,
        chat_fn=mock_chat_search,
    )
    assert res_case_fail.teacher_success is False
    assert len(res_case_fail.teacher_suffix) == 1
    assert res_case_fail.teacher_suffix[0].done is True
    assert res_case_fail.teacher_suffix[0].reward == 0.0

    # 4. Custom threshold (success_reward_threshold=0.5)
    engine_thresh = TeacherContinuationEngine(
        max_steps=5,
        max_retries=1,
        allow_simulated=False,
        success_reward_threshold=0.5,
    )

    # Sub-threshold reward (0.3 <= 0.5) must be rejected
    def mock_env_sub_thresh(action):
        return "Partial accomplishment", True, 0.3

    res_case_sub = engine_thresh.run_continuation(
        base_case,
        env_step_fn=mock_env_sub_thresh,
        chat_fn=mock_chat_search,
    )
    assert res_case_sub.teacher_success is False

    # Supra-threshold reward (0.8 > 0.5) must pass
    def mock_env_supra_thresh(action):
        return "High quality accomplishment", True, 0.8

    res_case_supra = engine_thresh.run_continuation(
        base_case,
        env_step_fn=mock_env_supra_thresh,
        chat_fn=mock_chat_search,
    )
    assert res_case_supra.teacher_success is True


def test_teacher_continuation_respects_env_done_on_finish_action():
    """Verify teacher FINISH action does NOT override env done=False to trigger false early success."""
    base_case = PairedSuffixCase(
        task_id="finish_env_done_task",
        task_description="Multi-step interactive environment task.",
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    def mock_chat_finish(system, user, **kw):
        return "<think>Premature completion</think><action>FINISH: done</action>", {}

    # Scenario A: Teacher emits FINISH, environment returns done=False with positive shaping reward.
    # Must NOT stop at step 1 and must NOT declare success!
    call_count_a = 0

    def mock_env_finish_not_done_positive_reward(action):
        nonlocal call_count_a
        call_count_a += 1
        return "Task still in progress (step not terminal)", False, 0.1

    engine_a = TeacherContinuationEngine(
        max_steps=3,
        max_retries=1,
        allow_simulated=False,
        success_reward_threshold=0.0,
    )

    res_case_a = engine_a.run_continuation(
        base_case,
        env_step_fn=mock_env_finish_not_done_positive_reward,
        chat_fn=mock_chat_finish,
    )

    assert res_case_a.teacher_success is False
    assert len(res_case_a.teacher_suffix) == 3
    assert call_count_a == 3
    for s in res_case_a.teacher_suffix:
        assert s.done is False
        assert s.reward == 0.1

    # Scenario B: Teacher emits FINISH, environment returns done=False with zero reward.
    call_count_b = 0

    def mock_env_finish_not_done_zero_reward(action):
        nonlocal call_count_b
        call_count_b += 1
        return "Task not finished", False, 0.0

    res_case_b = engine_a.run_continuation(
        base_case,
        env_step_fn=mock_env_finish_not_done_zero_reward,
        chat_fn=mock_chat_finish,
    )
    assert res_case_b.teacher_success is False
    assert len(res_case_b.teacher_suffix) == 3
    assert call_count_b == 3

    # Scenario C: Teacher emits FINISH, environment confirms true completion (done=True).
    def mock_env_finish_genuinely_done(action):
        return "Task successfully completed", True, 1.0

    res_case_c = engine_a.run_continuation(
        base_case,
        env_step_fn=mock_env_finish_genuinely_done,
        chat_fn=mock_chat_finish,
    )
    assert res_case_c.teacher_success is True
    assert len(res_case_c.teacher_suffix) == 1
    assert res_case_c.teacher_suffix[0].done is True


def test_skillpivot_teacher_max_steps_wiring_and_bounds():
    """Verify skillpivot_teacher_max_steps is wired through Pipeline and bounds teacher rollouts."""
    base_case = PairedSuffixCase(
        task_id="step_bound_task",
        task_description="Search and solve.",
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    # 1. Pipeline wiring check
    pipe_3 = SkillPivotPipeline(teacher_max_steps=3)
    assert pipe_3.teacher_engine.max_steps == 3

    pipe_7 = SkillPivotPipeline(teacher_max_steps=7)
    assert pipe_7.teacher_engine.max_steps == 7

    # 2. Execution bounds check with non-terminating environment
    def mock_chat_search(system, user, **kw):
        return "<think>Explore</think><action>read_item</action>", {}

    def mock_env_never_done(action):
        return "read result", False, 0.0

    res_3 = pipe_3.teacher_engine.run_continuation(
        base_case,
        env_step_fn=mock_env_never_done,
        chat_fn=mock_chat_search,
    )
    assert len(res_3.teacher_suffix) == 3
    assert res_3.teacher_success is False

    res_5 = TeacherContinuationEngine(max_steps=5, allow_simulated=False).run_continuation(
        base_case,
        env_step_fn=mock_env_never_done,
        chat_fn=mock_chat_search,
    )
    assert len(res_5.teacher_suffix) == 5
    assert res_5.teacher_success is False


def test_skillpivot_config_flattening_and_pipeline_wiring(tmp_path):
    """Verify optimizer.skillpivot_success_reward_threshold and teacher_max_steps flatten and propagate."""
    # 1. Verify searchqa/skillpivot.yaml flattens all SkillPivot keys properly
    yaml_path = "configs/searchqa/skillpivot.yaml"
    cfg = load_config(yaml_path)
    flat = flatten_config(cfg)

    assert "skillpivot_success_reward_threshold" in flat
    assert flat["skillpivot_success_reward_threshold"] == 0.0
    assert flat["skillpivot_teacher_max_steps"] == 10
    assert flat["skillpivot_gate_threshold"] == 0.5

    # 2. Verify structured config override flattening
    structured_cfg = {
        "optimizer": {
            "type": "skillpivot",
            "skillpivot_success_reward_threshold": 0.75,
            "skillpivot_teacher_max_steps": 4,
            "skillpivot_gate_threshold": 0.6,
        }
    }
    flat_custom = flatten_config(structured_cfg)
    assert flat_custom["skillpivot_success_reward_threshold"] == 0.75
    assert flat_custom["skillpivot_teacher_max_steps"] == 4
    assert flat_custom["skillpivot_gate_threshold"] == 0.6

    # 3. Verify trainer reflects teacher_max_steps and success_reward_threshold into pipeline
    tid = "task_config_prop"
    conversation = [
        {"role": "user", "content": "Question: Find the answer"},
        {"role": "assistant", "content": "<think>browse 1</think><action>browse</action>"},
        {"role": "user", "content": "Observation: error"},
        {"role": "assistant", "content": "<think>browse 2</think><action>browse</action>"},
        {"role": "user", "content": "Observation: error"},
    ]

    rollout_failed = [
        {
            "id": tid,
            "task_description": "An interactive environment task.",
            "ground_truth": "",
            "gold_answers": [],
            "hard": 0.0,
            "soft": 0.0,
        }
    ]

    pred_sub = tmp_path / "pred_sub"
    (pred_sub / tid).mkdir(parents=True, exist_ok=True)
    with open(pred_sub / tid / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    pred_supra = tmp_path / "pred_supra"
    (pred_supra / tid).mkdir(parents=True, exist_ok=True)
    with open(pred_supra / tid / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    # Sub-threshold reward (0.5 <= 0.75) must NOT succeed in teacher continuation
    def mock_env_sub(action):
        return "Sub threshold terminal", True, 0.5

    # Supra-threshold reward (0.9 > 0.75) with valid teacher should succeed continuation
    def mock_env_supra(action):
        return "Supra threshold terminal", True, 0.9

    def mock_chat_custom(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Solve</think><action>submit_solution</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Repeated browse error",
                "teacher_correction": "Extracted Gold Answer",
                "skill_gap": "Need planning rule",
                "evolution_direction": "Add planning before action",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Planning",
                "content": "Plan carefully before browsing.",
                "trigger_condition": "When receiving complex reasoning query",
                "rationale": "Prevents browse loop",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.9,
                "generalizable_rule_score": 0.9,
                "critique": "Solid improvement",
            }), {}
        return "", {}

    from skillopt.optimizer.skillpivot import (
        delta_generator,
        regression_verifier,
        teacher_continuation,
    )

    import unittest.mock
    with unittest.mock.patch.object(teacher_continuation, "chat_optimizer", side_effect=mock_chat_custom), \
         unittest.mock.patch.object(delta_generator, "chat_optimizer", side_effect=mock_chat_custom), \
         unittest.mock.patch.object(regression_verifier, "chat_optimizer", side_effect=mock_chat_custom):
        deltas_sub = run_skillpivot_reflect_step(
            rollout_results=rollout_failed,
            current_skill="# Base Skill",
            prediction_dir=str(pred_sub),
            cfg=flat_custom,
            env_step_fn=mock_env_sub,
            embedding_provider=MockEmbeddingProvider(),
            allow_simulated=False,
        )
        assert len(deltas_sub) == 0

        deltas_supra = run_skillpivot_reflect_step(
            rollout_results=rollout_failed,
            current_skill="# Base Skill",
            prediction_dir=str(pred_supra),
            cfg=flat_custom,
            env_step_fn=mock_env_supra,
            embedding_provider=MockEmbeddingProvider(),
            allow_simulated=False,
        )
        assert len(deltas_supra) == 1
        assert "Plan carefully before browsing" in deltas_supra[0]["patch"]["edits"][0]["content"]



