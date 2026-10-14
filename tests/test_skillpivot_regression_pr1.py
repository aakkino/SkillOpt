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


def test_teacher_continuation_without_env_step_allows_multiturn_finish_recovery():
    """Verify that absent env_step_fn, teacher tool action returns observation without done=True,
    allowing subsequent turns to recover and emit FINISH: [gold] to succeed continuation.

    Regression test for round 6 review P1 finding:
    PairedSuffixCase(ground_truth='4'), teacher turn 1 outputs <action>search docs</action>,
    teacher turn 2 outputs <action>FINISH: 4</action>.
    Must NOT terminate after turn 1; must succeed at turn 2 with 2 teacher steps.
    """
    engine = TeacherContinuationEngine(
        max_steps=5,
        max_retries=1,
        allow_simulated=False,
    )

    case = PairedSuffixCase(
        task_id="recovery_task_4",
        task_description="Solve the puzzle and find the number.",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    turn_count = 0

    def mock_chat_multiturn(system, user, **kw):
        nonlocal turn_count
        turn_count += 1
        if turn_count == 1:
            return "<think>Need docs</think><action>search docs</action>", {}
        else:
            return "<think>Got it, submit final answer</think><action>FINISH: 4</action>", {}

    res = engine.run_continuation(
        case,
        env_step_fn=None,
        chat_fn=mock_chat_multiturn,
    )

    assert res.teacher_success is True
    assert len(res.teacher_suffix) == 2
    assert turn_count == 2
    # Turn 1: tool action without env_step_fn
    assert res.teacher_suffix[0].action == "search docs"
    assert "No environment step function available" in res.teacher_suffix[0].observation
    assert res.teacher_suffix[0].done is False
    assert res.teacher_suffix[0].reward == 0.0
    assert res.teacher_suffix[0].execution_status == 0
    # Turn 2: recovered finish action
    assert "FINISH: 4" in res.teacher_suffix[1].action
    assert res.teacher_suffix[1].done is True
    assert res.teacher_suffix[1].reward == 1.0
    assert res.teacher_suffix[1].execution_status == 1


def test_trainer_pipeline_adapter_continuation_step_fn_wiring(tmp_path, monkeypatch):
    """Verify that Trainer invokes run_skillpivot_reflect_step with adapter,
    and Pipeline automatically acquires per-task continuation step function from adapter.build_continuation_step_fn."""
    from skillopt.envs.base import EnvAdapter
    from tests.test_deviation_detector import MockEmbeddingProvider

    class MockInteractiveAdapter(EnvAdapter):
        def __init__(self):
            super().__init__()
            self.step_history = []

        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["interactive"]

        def build_continuation_step_fn(self, case, **kwargs):
            # Returns a callable that steps through 2 tool actions before terminal success
            steps_taken = 0

            def _step_fn(action: str):
                nonlocal steps_taken
                steps_taken += 1
                self.step_history.append((case.task_id, action, steps_taken))
                if steps_taken < 2:
                    return f"Obs for {action}", False, 0.1
                return f"Final Obs for {action}", True, 1.0

            return _step_fn

    adapter = MockInteractiveAdapter()

    pred_dir = tmp_path / "pred_adapter_interactive"
    tid = "task_interactive_01"
    (pred_dir / tid).mkdir(parents=True, exist_ok=True)
    conversation = [
        {"step": 1, "action": "open door", "env_feedback": "door open", "execution_status": 1},
        {"step": 2, "action": "wrong turn", "env_feedback": "blocked", "execution_status": 0},
    ]
    with open(pred_dir / tid / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    rollout_failed = [
        {
            "id": tid,
            "task_description": "Navigate to the goal.",
            "ground_truth": "",
            "gold_answers": [],
            "hard": 0.0,
            "soft": 0.0,
        }
    ]

    action_count = 0

    def mock_chat_interactive(system, user, stage="", **kw):
        nonlocal action_count
        if stage == "teacher_continuation":
            action_count += 1
            if action_count == 1:
                return "<think>Step 1</think><action>take tool step 1</action>", {}
            else:
                return "<think>Step 2</think><action>take tool step 2</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Wrong turn",
                "teacher_correction": "Follow path",
                "skill_gap": "Need navigation rule",
                "evolution_direction": "Add navigation rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Navigation",
                "content": "Verify door state before moving.",
                "trigger_condition": "When doorway is encountered",
                "rationale": "Prevents wrong path",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.9,
                "generalizable_rule_score": 0.9,
                "critique": "Solid update",
            }), {}
        return "", {}

    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_interactive)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_interactive)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_interactive)

    cfg = {
        "optimizer_type": "skillpivot",
        "skillpivot_embedding_backend": "local",
        "skillpivot_gate_threshold": 0.5,
        "skillpivot_teacher_max_steps": 5,
        "skillpivot_success_reward_threshold": 0.0,
    }

    raw_patches = run_skillpivot_reflect_step(
        rollout_results=rollout_failed,
        current_skill="# Navigation Skill",
        prediction_dir=str(pred_dir),
        cfg=cfg,
        adapter=adapter,
        embedding_provider=MockEmbeddingProvider(),
        allow_simulated=False,
    )

    assert len(raw_patches) == 1
    assert "Verify door state before moving." in raw_patches[0]["patch"]["edits"][0]["content"]
    assert len(adapter.step_history) == 2
    assert adapter.step_history[0] == (tid, "take tool step 1", 1)
    assert adapter.step_history[1] == (tid, "take tool step 2", 2)


def test_searchqa_adapter_continuation_hooks_contract():
    """Verify SearchQAAdapter continuation hooks adhere to single-turn QA contract."""
    from skillopt.envs.searchqa.adapter import SearchQAAdapter

    sqa_adapter = SearchQAAdapter()
    assert sqa_adapter.build_continuation_step_fn(None) is None

    eval_fn = sqa_adapter.build_continuation_evaluator_fn(None)
    assert callable(eval_fn)
    # Exact match passes
    assert eval_fn("Albert Einstein", ["Albert Einstein", "Einstein"]) is True
    assert eval_fn("Einstein", ["Albert Einstein", "Einstein"]) is True
    # Sub-exact match fails
    assert eval_fn("40", ["4"]) is False
    assert eval_fn("", ["Einstein"]) is False


# ── 15. Round 7 Review Regressions: train_env Context & Teacher Retry Reconstruction ───────


def test_trainer_passes_train_env_context_to_adapter_continuation_hooks(tmp_path, monkeypatch):
    """Verify that Trainer passes train_env context through run_skillpivot_reflect_step
    to pipeline and adapter.build_continuation_step_fn / build_continuation_evaluator_fn."""
    from skillopt.envs.base import EnvAdapter
    from tests.test_deviation_detector import MockEmbeddingProvider

    received_contexts = {
        "step_fn_train_env": None,
        "eval_fn_train_env": None,
    }

    mock_train_env_instance = {
        "env_type": "batch_docker_sim",
        "checkpoint_epoch": 2,
        "active_workers": 4,
    }

    class ContextCheckingAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["interactive"]

        def build_continuation_step_fn(self, case, train_env=None, attempt=1, **kwargs):
            received_contexts["step_fn_train_env"] = train_env
            # Simple step function returning success
            return lambda action: (f"Result for {action}", True, 1.0)

        def build_continuation_evaluator_fn(self, case, train_env=None, **kwargs):
            received_contexts["eval_fn_train_env"] = train_env
            return lambda pred, gold: True

    adapter = ContextCheckingAdapter()

    pred_dir = tmp_path / "pred_context_check"
    tid = "task_ctx_01"
    (pred_dir / tid).mkdir(parents=True, exist_ok=True)
    conversation = [
        {"step": 1, "action": "inspect item", "env_feedback": "error state", "execution_status": 0},
    ]
    with open(pred_dir / tid / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    rollout_failed = [
        {
            "id": tid,
            "task_description": "Context check task.",
            "ground_truth": "",
            "gold_answers": [],
            "hard": 0.0,
            "soft": 0.0,
        }
    ]

    def mock_chat(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Fix</think><action>do tool action</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed inspection",
                "teacher_correction": "Inspect correctly",
                "skill_gap": "Need inspection rule",
                "evolution_direction": "Add inspection rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Inspection",
                "content": "Always verify tool parameters before inspection.",
                "trigger_condition": "During inspection",
                "rationale": "Ensures success",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Valid inspection rule",
            }), {}
        return "", {}

    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat)

    cfg = {
        "optimizer_type": "skillpivot",
        "skillpivot_embedding_backend": "local",
        "skillpivot_gate_threshold": 0.5,
        "skillpivot_teacher_max_steps": 3,
        "skillpivot_success_reward_threshold": 0.0,
    }

    raw_patches = run_skillpivot_reflect_step(
        rollout_results=rollout_failed,
        current_skill="# Base Skill",
        prediction_dir=str(pred_dir),
        cfg=cfg,
        adapter=adapter,
        embedding_provider=MockEmbeddingProvider(),
        allow_simulated=False,
        train_env=mock_train_env_instance,
    )

    assert len(raw_patches) == 1
    # Verify train_env was strictly and accurately delivered to both hooks
    assert received_contexts["step_fn_train_env"] is mock_train_env_instance
    assert received_contexts["eval_fn_train_env"] is mock_train_env_instance


def test_teacher_continuation_retries_with_stateful_environment_reconstruction():
    """Verify that when attempt 1 ends with done=True, reward=0 (failure termination),
    TeacherContinuationEngine rebuilds / resets stateful environment for attempt 2,
    allowing successful continuation from the same deviation point t* without encountering
    'Episode already terminated' error.

    Regression test for round 7 review P1 finding:
    Stateful adapter where attempt 1 action terminates with failure, and subsequent attempt 2
    action must run on fresh/reset state to succeed.
    """
    class StatefulEnvironment:
        def __init__(self, task_id: str):
            self.task_id = task_id
            self.is_terminated = False
            self.actions_received = []

        def step(self, action: str):
            self.actions_received.append(action)
            if self.is_terminated:
                # Real stateful environments reject actions after terminal status
                return "Episode already terminated: invalid action after done=True", True, 0.0

            if action == "wrong action":
                self.is_terminated = True
                return "Trap triggered! Mission failed.", True, 0.0
            elif action == "correct action":
                self.is_terminated = True
                return "Key turned, door opened successfully!", True, 1.0
            else:
                return f"Obs: {action}", False, 0.0

    from skillopt.envs.base import EnvAdapter

    class StatefulAdapter(EnvAdapter):
        def __init__(self):
            super().__init__()
            self.envs_created = []

        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["stateful"]

        def build_continuation_step_fn(self, case, train_env=None, attempt=1, **kwargs):
            # Construct a brand new stateful environment per attempt from deviation point t*
            env = StatefulEnvironment(case.task_id)
            self.envs_created.append((attempt, env))
            return env.step

    adapter = StatefulAdapter()

    engine = TeacherContinuationEngine(
        max_steps=5,
        max_retries=2,
        allow_simulated=False,
        success_reward_threshold=0.0,
    )

    case = PairedSuffixCase(
        task_id="task_stateful_door",
        task_description="Open the locked door safely.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0

    def mock_chat_retry(system, user, **kw):
        nonlocal attempt_counter
        # On attempt 1, model outputs wrong action leading to termination
        # On attempt 2, model outputs correct action leading to success
        if attempt_counter == 0:
            attempt_counter += 1
            return "<think>Try force</think><action>wrong action</action>", {}
        else:
            attempt_counter += 1
            return "<think>Try right key</think><action>correct action</action>", {}

    res = engine.run_continuation(
        case,
        chat_fn=mock_chat_retry,
        adapter=adapter,
    )

    assert res.teacher_success is True
    # Verify that two attempts took place and two environment instances were created
    assert len(adapter.envs_created) == 2
    assert adapter.envs_created[0][0] == 1
    assert adapter.envs_created[1][0] == 2

    env_attempt1 = adapter.envs_created[0][1]
    env_attempt2 = adapter.envs_created[1][1]

    # Attempt 1 failed with trap
    assert env_attempt1.actions_received == ["wrong action"]
    assert env_attempt1.is_terminated is True

    # Attempt 2 ran on fresh environment from t* and succeeded
    assert env_attempt2.actions_received == ["correct action"]
    assert env_attempt2.is_terminated is True

    # Verify final teacher suffix reflects successful attempt 2
    assert len(res.teacher_suffix) == 1
    assert res.teacher_suffix[0].action == "correct action"
    assert "door opened successfully" in res.teacher_suffix[0].observation
    assert res.teacher_suffix[0].done is True
    assert res.teacher_suffix[0].reward == 1.0
    assert res.teacher_suffix[0].execution_status == 1


# ── 15. Round 8 Regression: Pipeline In-situ Reset Retry & Exception Logging ──


def test_pipeline_teacher_continuation_in_situ_reset_retry(monkeypatch):
    """Verify that when running through the real SkillPivotPipeline.optimize_step,
    an adapter providing in-situ environment reset (reset_continuation_env) has its reset
    hook reliably invoked before retry attempt 2, allowing successful teacher continuation
    without remaining in the terminal state.

    Regression test for round 8 review P1 finding:
    Prior code passed env_step_fn_builder to run_continuation, causing the engine's elif adapter
    reset branch to be bypassed and leaving reset_calls empty [].
    """
    from skillopt.envs.base import EnvAdapter
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class InSituResetAdapter(EnvAdapter):
        def __init__(self):
            super().__init__()
            self.is_terminated = False
            self.actions_received = []
            self.reset_calls = []

        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["insitu"]

        def step(self, action: str):
            self.actions_received.append(action)
            if self.is_terminated:
                return "Episode already terminated: invalid action after done=True", True, 0.0

            if action == "bad step":
                self.is_terminated = True
                return "Failed step", True, 0.0
            elif action == "good step":
                self.is_terminated = True
                return "Goal achieved successfully!", True, 1.0
            return "Progress", False, 0.0

        def build_continuation_step_fn(self, case, train_env=None, attempt=1, **kwargs):
            return self.step

        def reset_continuation_env(self, case, train_env=None, attempt=1, **kwargs):
            self.reset_calls.append(attempt)
            self.is_terminated = False

    adapter = InSituResetAdapter()

    attempt_counter = 0

    def mock_chat_retry_pipeline(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            if attempt_counter == 0:
                attempt_counter += 1
                return "<think>Try initial idea</think><action>bad step</action>", {}
            else:
                attempt_counter += 1
                return "<think>Correct approach</think><action>good step</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed step",
                "teacher_correction": "Use good step",
                "skill_gap": "Need good step guideline",
                "evolution_direction": "Add rule for good step",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Step Choice",
                "content": "Always choose good step over bad step.",
                "trigger_condition": "When choosing steps",
                "rationale": "Leads to success",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Solid update",
            }), {}
        return "", {}

    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_retry_pipeline)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_retry_pipeline)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_retry_pipeline)

    pipeline = SkillPivotPipeline(
        allow_simulated=False,
        teacher_max_steps=5,
    )
    pipeline.teacher_engine.max_retries = 2

    case = PairedSuffixCase(
        task_id="task_insitu_pipeline",
        task_description="Execute good step to accomplish the goal.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill, accepted_deltas, gate_results = pipeline.optimize_step(
        paired_cases=[case],
        current_skill="# Initial Skill",
        adapter=adapter,
    )

    # Verify teacher succeeded on attempt 2
    assert case.teacher_success is True
    # Verify in-situ reset hook was called exactly once for attempt 2
    assert adapter.reset_calls == [2]
    assert adapter.actions_received == ["bad step", "good step"]
    assert len(accepted_deltas) == 1
    assert "Always choose good step" in updated_skill

    # Also verify direct engine call with the same in-situ adapter
    adapter_direct = InSituResetAdapter()
    engine = pipeline.teacher_engine
    attempt_counter = 0
    case_direct = PairedSuffixCase(
        task_id="task_insitu_direct",
        task_description="Execute good step directly.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )
    res_direct = engine.run_continuation(
        case_direct,
        chat_fn=mock_chat_retry_pipeline,
        adapter=adapter_direct,
    )
    assert res_direct.teacher_success is True
    assert adapter_direct.reset_calls == [2]
    assert adapter_direct.actions_received == ["bad step", "good step"]


def test_teacher_continuation_builder_and_adapter_exception_logging(caplog):
    """Verify that when env_step_fn_builder or adapter raises an exception during environment
    reconstruction/reset, the error is safely caught and logged without triggering NameError,
    and crucially, environment construction failure is NOT silently converted to a successful
    answer verification (teacher_success remains False, preventing fake successful suffixes).

    Regression test for round 8 review P2 finding & round 9 review P1 finding.
    """
    import logging

    engine = TeacherContinuationEngine(
        max_steps=3,
        max_retries=2,
        allow_simulated=False,
    )

    case = PairedSuffixCase(
        task_id="task_exception_logger",
        task_description="Task with failing builder.",
        ground_truth="target_42",
        gold_answers=["target_42"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    # 1. env_step_fn_builder raises RuntimeError
    def failing_builder(attempt: int):
        raise RuntimeError("checkpoint unavailable")

    def mock_chat_answer(system, user, **kw):
        return "<think>Done</think><action>FINISH: target_42</action>", {}

    with caplog.at_level(logging.WARNING):
        res = engine.run_continuation(
            case,
            env_step_fn_builder=failing_builder,
            chat_fn=mock_chat_answer,
        )

    # Verify no NameError was raised and failure state was preserved
    assert res.teacher_success is False
    assert any("env_step_fn_builder error on attempt 1: checkpoint unavailable" in r.message for r in caplog.records)
    assert any("env_step_fn_builder error on attempt 2: checkpoint unavailable" in r.message for r in caplog.records)
    assert len(res.teacher_suffix) == 1
    assert res.teacher_suffix[0].action == "ENVIRONMENT_ERROR"
    assert "checkpoint unavailable" in res.teacher_suffix[0].observation
    assert res.teacher_suffix[0].execution_status == 0

    caplog.clear()

    # 2. adapter.build_continuation_step_fn raises RuntimeError
    class FailingAdapter:
        def build_continuation_step_fn(self, case, **kwargs):
            raise RuntimeError("adapter rebuild failed")

    with caplog.at_level(logging.WARNING):
        res2 = engine.run_continuation(
            case,
            adapter=FailingAdapter(),
            chat_fn=mock_chat_answer,
        )

    assert res2.teacher_success is False
    assert any("failed to rebuild step fn on attempt 1: adapter rebuild failed" in r.message for r in caplog.records)
    assert len(res2.teacher_suffix) == 1
    assert res2.teacher_suffix[0].action == "ENVIRONMENT_ERROR"
    assert "adapter rebuild failed" in res2.teacher_suffix[0].observation

    caplog.clear()

    # 3. adapter.reset_continuation_env raises RuntimeError on attempt 2
    class FailingResetAdapter:
        def __init__(self):
            self.step_called = 0

        def build_continuation_step_fn(self, case, **kwargs):
            def _step(action):
                self.step_called += 1
                if self.step_called == 1:
                    return "First attempt failed", True, 0.0
                return "Second attempt succeeded", True, 1.0
            return _step

        def reset_continuation_env(self, case, **kwargs):
            raise RuntimeError("reset environment failed")

    def mock_chat_two_steps(system, user, **kw):
        return "<think>Act</think><action>do something</action>", {}

    with caplog.at_level(logging.WARNING):
        res3 = engine.run_continuation(
            case,
            adapter=FailingResetAdapter(),
            chat_fn=mock_chat_two_steps,
        )

    # Verify no NameError was raised and reset warning logged
    assert any("adapter.reset_continuation_env error on attempt 2: reset environment failed" in r.message for r in caplog.records)


# ── 16. Round 9 Regression: Environment Construction Failure vs Explicit QA Mode ──


def test_environment_construction_failure_rejected_in_pipeline_while_searchqa_succeeds(monkeypatch):
    """Verify that when an interactive environment adapter fails to build continuation step function
    (e.g. raises RuntimeError), the pipeline preserves the failure status, rejects the attempt,
    and skips delta generation / skill update, even if gold_answers are present and the teacher
    outputs a matching answer.

    Conversely, when an adapter explicitly declares stateless QA takeover by returning None
    (e.g. SearchQAAdapter), pure answer verification against gold answers remains supported and succeeds.

    Regression test for round 9 review P1 finding.
    """
    from skillopt.envs.base import EnvAdapter
    from skillopt.envs.searchqa.adapter import SearchQAAdapter
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    # ── 1. Interactive Adapter with Construction Exception: Must be rejected ──
    class InteractiveFailingAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["interactive"]

        def build_continuation_step_fn(self, case, **kwargs):
            raise RuntimeError("checkpoint unavailable")

    def mock_chat_interactive_bypass(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Bypass tool</think><action>FINISH: 4</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_interactive_bypass)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_interactive_bypass)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_interactive_bypass)

    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=5)
    pipeline.teacher_engine.max_retries = 2

    interactive_case = PairedSuffixCase(
        task_id="task_interactive_fail",
        task_description="Execute interactive tools to compute answer.",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill, accepted_deltas, gate_results = pipeline.optimize_step(
        paired_cases=[interactive_case],
        current_skill="# Initial Skill",
        adapter=InteractiveFailingAdapter(),
    )

    # Verification: teacher must NOT succeed, no suffix accepted, no delta generated
    assert interactive_case.teacher_success is False
    assert len(accepted_deltas) == 0
    assert len(gate_results) == 0
    assert updated_skill == "# Initial Skill"
    assert len(interactive_case.teacher_suffix) == 1
    assert interactive_case.teacher_suffix[0].action == "ENVIRONMENT_ERROR"
    assert "checkpoint unavailable" in interactive_case.teacher_suffix[0].observation

    # ── 2. SearchQA Adapter Explicit Stateless QA: Must succeed on answer match ──
    searchqa_adapter = SearchQAAdapter(max_turns=3)

    def mock_chat_searchqa(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Exact answer</think><action>FINISH: 4</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Wrong calculation",
                "teacher_correction": "Answer 4",
                "skill_gap": "Need accurate answer rule",
                "evolution_direction": "Add rule for 4",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Calculation",
                "content": "Verify calculation yields 4.",
                "trigger_condition": "When calculating",
                "rationale": "Matches ground truth",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Correct QA update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_searchqa)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_searchqa)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_searchqa)

    qa_case = PairedSuffixCase(
        task_id="task_searchqa_normal",
        task_description="What is 2 + 2?",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill_qa, accepted_deltas_qa, gate_results_qa = pipeline.optimize_step(
        paired_cases=[qa_case],
        current_skill="# Initial Skill",
        adapter=searchqa_adapter,
    )

    # Verification: SearchQA explicit QA mode succeeds normally
    assert qa_case.teacher_success is True
    assert len(accepted_deltas_qa) == 1
    assert "Verify calculation yields 4." in updated_skill_qa
    assert len(qa_case.teacher_suffix) == 1
    assert qa_case.teacher_suffix[0].action == "FINISH: 4"
    assert qa_case.teacher_suffix[0].observation == "Answer verified: OK"
    assert qa_case.teacher_suffix[0].done is True
    assert qa_case.teacher_suffix[0].reward == 1.0


# ── 17. Round 9 Regression: Bound Step/Reset Retry Without Adapter ────────────


def test_pipeline_stateful_bound_env_step_reset_retry_without_adapter(monkeypatch):
    """Verify that when running through SkillPivotPipeline.optimize_step with env_step_fn=env.step
    and NO adapter (or an adapter without reset_continuation_env override), the pipeline and engine
    correctly invoke the bound env.reset method before retry attempt 2 rather than falsely treating
    a no-op reset callback as having reset the environment.

    Regression test for round 9 review P2 finding.
    """
    from skillopt.envs.base import EnvAdapter
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class StatefulEnvWithoutAdapter:
        def __init__(self):
            self.is_terminated = False
            self.actions_received = []
            self.reset_calls = []

        def reset(self, attempt=1):
            self.reset_calls.append(attempt)
            self.is_terminated = False

        def step(self, action: str):
            self.actions_received.append(action)
            if self.is_terminated:
                return "Episode already terminated: invalid action after done=True", True, 0.0

            if action == "wrong action":
                self.is_terminated = True
                return "Failed on wrong step", True, 0.0
            elif action == "correct action":
                self.is_terminated = True
                return "Task completed successfully!", True, 1.0
            return "Progress", False, 0.0

    attempt_counter = 0

    def mock_chat_bound_retry(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            if attempt_counter == 0:
                attempt_counter += 1
                return "<think>Try initial action</think><action>wrong action</action>", {}
            else:
                attempt_counter += 1
                return "<think>Try corrected action</think><action>correct action</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed action",
                "teacher_correction": "Use correct action",
                "skill_gap": "Need rule",
                "evolution_direction": "Add rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Action Choice",
                "content": "Always choose correct action.",
                "trigger_condition": "When choosing",
                "rationale": "Leads to success",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Good update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_bound_retry)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_bound_retry)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_bound_retry)

    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=5)
    pipeline.teacher_engine.max_retries = 2

    # ── Branch A: env_step_fn=env.step directly without adapter ──
    env_direct = StatefulEnvWithoutAdapter()
    case_direct = PairedSuffixCase(
        task_id="task_bound_direct",
        task_description="Perform action to complete stateful task.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_dir, accepted_deltas_dir, _ = pipeline.optimize_step(
        paired_cases=[case_direct],
        current_skill="# Initial Skill",
        env_step_fn=env_direct.step,
        adapter=None,
    )

    assert case_direct.teacher_success is True
    assert env_direct.reset_calls == [2]
    assert env_direct.actions_received == ["wrong action", "correct action"]
    assert len(accepted_deltas_dir) == 1
    assert "Always choose correct action." in updated_skill_dir

    # ── Branch B: adapter inherits EnvAdapter but does NOT override reset_continuation_env ──
    class BaseNoopAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["noop_reset"]

        def build_continuation_step_fn(self, case, **kwargs):
            return env_noop.step

    env_noop = StatefulEnvWithoutAdapter()
    case_noop = PairedSuffixCase(
        task_id="task_bound_noop_adapter",
        task_description="Perform action with base adapter.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_noop, accepted_deltas_noop, _ = pipeline.optimize_step(
        paired_cases=[case_noop],
        current_skill="# Initial Skill",
        adapter=BaseNoopAdapter(),
    )

    assert case_noop.teacher_success is True
    assert env_noop.reset_calls == [2]
    assert env_noop.actions_received == ["wrong action", "correct action"]
    assert len(accepted_deltas_noop) == 1
    assert "Always choose correct action." in updated_skill_noop


# ── 18. Round 10 Regression: Tool Step Execution Exception Gating ─────────────


def test_tool_execution_exception_rejected_in_pipeline_even_with_matching_answer(monkeypatch):
    """Verify that when active_env_step_fn raises an execution exception (e.g. RuntimeError),
    the attempt is marked as failed and is NEVER accepted via answer matching even if the
    teacher outputs a matching gold answer. Delta generation and skill update must be skipped.

    Regression test for round 10 review P1 finding.
    """
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class ToolEnvWithReset:
        def __init__(self):
            self.reset_calls = []

        def reset(self, attempt=1):
            self.reset_calls.append(attempt)

        def step(self, action: str):
            raise RuntimeError("tool server offline")

    def mock_chat_step_error(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Direct answer</think><action>FINISH: 4</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_step_error)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_step_error)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_step_error)

    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=3)
    pipeline.teacher_engine.max_retries = 2

    case = PairedSuffixCase(
        task_id="task_tool_exec_error",
        task_description="Execute tool to obtain answer.",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    env = ToolEnvWithReset()
    updated_skill, accepted_deltas, gate_results = pipeline.optimize_step(
        paired_cases=[case],
        current_skill="# Initial Skill",
        env_step_fn=env.step,
        adapter=None,
    )

    # Verification: teacher must NOT succeed; execution error observation recorded; no delta generated
    assert case.teacher_success is False
    assert len(accepted_deltas) == 0
    assert len(gate_results) == 0
    assert updated_skill == "# Initial Skill"
    assert len(case.teacher_suffix) >= 1
    last_step = case.teacher_suffix[-1]
    assert "Execution error: tool server offline" in last_step.observation
    assert last_step.reward == 0.0
    assert last_step.execution_status == 0


# ── 19. Round 10 Regression: Callable Factory Exception Gating ────────────────


def test_callable_factory_exception_rejected_without_fallback_in_pipeline(monkeypatch):
    """Verify that when a callable factory `def build(case, attempt)` raises an exception,
    the pipeline does NOT fallback to returning the factory itself (which would cause a TypeError
    and subsequent false-positive answer matching). Instead, the construction failure is gated
    and delta generation is skipped.

    Regression test for round 10 review P1 finding.
    """
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    def failing_factory(case, attempt):
        raise RuntimeError("database connection lost")

    def mock_chat_factory_error(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Direct answer</think><action>FINISH: 4</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_factory_error)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_factory_error)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_factory_error)

    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=3)
    pipeline.teacher_engine.max_retries = 2

    case = PairedSuffixCase(
        task_id="task_factory_error",
        task_description="Build tool step fn via factory.",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill, accepted_deltas, _ = pipeline.optimize_step(
        paired_cases=[case],
        current_skill="# Initial Skill",
        env_step_fn=failing_factory,
        adapter=None,
    )

    # Verification: teacher must NOT succeed; ENVIRONMENT_ERROR recorded; no delta generated
    assert case.teacher_success is False
    assert len(accepted_deltas) == 0
    assert updated_skill == "# Initial Skill"
    assert len(case.teacher_suffix) == 1
    assert case.teacher_suffix[0].action == "ENVIRONMENT_ERROR"
    assert "database connection lost" in case.teacher_suffix[0].observation


# ── 20. Round 10 Regression: Reset Failure & State Pollution Suffix Rejection ──


def test_reset_failure_rejects_polluted_state_suffix_while_clean_builder_succeeds(monkeypatch):
    """Verify that when a stateful environment fails reset on attempt > 1 (raising an exception
    or returning False), the engine rejects the retry attempt rather than continuing with
    polluted state changes from attempt 1.

    Conversely, a clean builder that independently restores/creates state succeeds on retry.

    Regression test for round 10 review P1 finding.
    """
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    # ── Part A: Stateful environment with failing reset (raises RuntimeError) ──
    class StatefulEnvWithFailingReset:
        def __init__(self, fail_with_exception=True):
            self.door_open = False
            self.fail_with_exception = fail_with_exception
            self.actions_received = []

        def reset(self, attempt=1):
            if self.fail_with_exception:
                raise RuntimeError("restore checkpoint failed")
            return False

        def step(self, action: str):
            self.actions_received.append(action)
            if action == "open":
                self.door_open = True
                return "Door opened", False, 0.0
            elif action == "collect":
                if self.door_open:
                    return "Collected item behind open door!", True, 1.0
                return "Door closed, cannot collect", True, 0.0
            return "Unknown action", False, 0.0

    attempt_counter = 0

    def mock_chat_two_step(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            if attempt_counter == 0:
                attempt_counter += 1
                return "<think>Open the door</think><action>open</action>", {}
            else:
                attempt_counter += 1
                return "<think>Collect item</think><action>collect</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_two_step)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_two_step)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_two_step)

    # Subcase A1: reset raises RuntimeError
    env_fail_exc = StatefulEnvWithFailingReset(fail_with_exception=True)
    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline.teacher_engine.max_retries = 2

    case_fail_exc = PairedSuffixCase(
        task_id="task_reset_fail_exc",
        task_description="Open door then collect.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_exc, accepted_deltas_exc, _ = pipeline.optimize_step(
        paired_cases=[case_fail_exc],
        current_skill="# Initial Skill",
        env_step_fn=env_fail_exc.step,
        adapter=None,
    )

    # Must NOT accept collect as a standalone suffix relying on attempt 1 open side-effect
    assert case_fail_exc.teacher_success is False
    assert len(accepted_deltas_exc) == 0
    assert updated_skill_exc == "# Initial Skill"
    assert env_fail_exc.actions_received == ["open"]

    # Subcase A2: reset returns False
    env_fail_false = StatefulEnvWithFailingReset(fail_with_exception=False)
    case_fail_false = PairedSuffixCase(
        task_id="task_reset_fail_false",
        task_description="Open door then collect.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_false, accepted_deltas_false, _ = pipeline.optimize_step(
        paired_cases=[case_fail_false],
        current_skill="# Initial Skill",
        env_step_fn=env_fail_false.step,
        adapter=None,
    )

    assert case_fail_false.teacher_success is False
    assert len(accepted_deltas_false) == 0
    assert updated_skill_false == "# Initial Skill"
    assert env_fail_false.actions_received == ["open"]

    # ── Part B: Independent Clean Builder Reconstructing State: Must Succeed ──
    class CleanStatefulEnv:
        def __init__(self):
            self.door_open = False

        def step(self, action: str):
            if action == "wrong":
                return "Wrong action", True, 0.0
            elif action == "open and collect":
                return "Completed in single step", True, 1.0
            return "Noop", False, 0.0

    def clean_builder(attempt: int = 1):
        env = CleanStatefulEnv()
        return env.step

    attempt_counter = 0

    def mock_chat_clean_builder(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            if attempt_counter == 0:
                attempt_counter += 1
                return "<think>Try wrong</think><action>wrong</action>", {}
            else:
                attempt_counter += 1
                return "<think>Try full sequence</think><action>open and collect</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Wrong step",
                "teacher_correction": "Open and collect",
                "skill_gap": "Need complete rule",
                "evolution_direction": "Add rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Complete Sequence",
                "content": "Perform complete open and collect sequence.",
                "trigger_condition": "When collecting",
                "rationale": "Succeeds cleanly",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Solid update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_clean_builder)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_clean_builder)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_clean_builder)

    pipeline_clean = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=2)
    pipeline_clean.teacher_engine.max_retries = 2

    case_clean = PairedSuffixCase(
        task_id="task_clean_builder_retry",
        task_description="Clean builder retry test.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_clean, accepted_deltas_clean, _ = pipeline_clean.optimize_step(
        paired_cases=[case_clean],
        current_skill="# Initial Skill",
        env_step_fn_builder=clean_builder,
        adapter=None,
    )

    # Verification: clean builder retry succeeds and generates delta
    assert case_clean.teacher_success is True
    assert len(accepted_deltas_clean) == 1
    assert "Perform complete open and collect sequence." in updated_skill_clean


def test_skipped_attempts_persist_environment_pollution_across_three_or_more_retries(monkeypatch):
    """Verify that when attempts are skipped (due to RESET_ERROR or ENVIRONMENT_ERROR),
    the pollution record of a stateful environment that executed steps is NOT cleared.
    Under the default max_retries=3 (and even max_retries=4), attempts 2 and 3 must both
    be rejected if the same unreset environment is passed, preventing attempt 3 from
    falsely succeeding on a polluted suffix.

    Regression test for round 11 review P1 finding.
    """
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class StatefulEnvWithoutReset:
        def __init__(self):
            self.door_open = False
            self.actions_received = []

        def step(self, action: str):
            self.actions_received.append(action)
            if action == "open":
                self.door_open = True
                return "Door opened", False, 0.0
            elif action == "collect":
                if self.door_open:
                    return "Collected item behind open door!", True, 1.0
                return "Door closed, cannot collect", True, 0.0
            return "Unknown action", False, 0.0

    attempt_counter = 0

    def mock_chat_multi_retry(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            attempt_counter += 1
            if attempt_counter == 1:
                return "<think>Open door</think><action>open</action>", {}
            else:
                return "<think>Collect item</think><action>collect</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_multi_retry)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_multi_retry)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_multi_retry)

    # 1. 默认 max_retries=3 场景
    env_default_retries = StatefulEnvWithoutReset()
    pipeline_3 = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    assert pipeline_3.teacher_engine.max_retries == 3

    case_3 = PairedSuffixCase(
        task_id="task_p1_multi_retry_default",
        task_description="Open door then collect.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_3, accepted_deltas_3, _ = pipeline_3.optimize_step(
        paired_cases=[case_3],
        current_skill="# Initial Skill",
        env_step_fn=env_default_retries.step,
        adapter=None,
    )

    # 断言：旧环境污染状态持续保留，attempt 2 与 attempt 3 均被门控拦截；
    # collect 绝未被执行，更不能被误判为 teacher_success
    assert case_3.teacher_success is False
    assert len(accepted_deltas_3) == 0
    assert updated_skill_3 == "# Initial Skill"
    assert env_default_retries.actions_received == ["open"]

    # 2. 4 次重试且中间穿插构造失败的复杂生命周期场景
    env_4 = StatefulEnvWithoutReset()
    pipeline_4 = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline_4.teacher_engine.max_retries = 4

    builder_calls = []

    def builder_with_crashes(attempt: int = 1):
        builder_calls.append(attempt)
        if attempt == 1:
            return env_4.step
        elif attempt == 2:
            raise RuntimeError("transient builder network error")
        elif attempt == 3:
            return env_4.step  # 依然返回受污染的未 reset 环境
        else:
            return env_4.step

    case_4 = PairedSuffixCase(
        task_id="task_p1_multi_retry_with_builder_crash",
        task_description="Open door then collect across builder crash.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_4, accepted_deltas_4, _ = pipeline_4.optimize_step(
        paired_cases=[case_4],
        current_skill="# Initial Skill",
        env_step_fn_builder=builder_with_crashes,
        adapter=None,
    )

    assert case_4.teacher_success is False
    assert len(accepted_deltas_4) == 0
    assert updated_skill_4 == "# Initial Skill"
    assert env_4.actions_received == ["open"]
    assert builder_calls == [1, 2, 3, 4]


def test_old_environment_reset_failure_does_not_reject_clean_independent_builder_retry(monkeypatch):
    """Verify that when an old environment fails to reset (via exception or returning False),
    an independent clean builder that returns a fresh, unpolluted environment succeeds
    and produces a valid skill delta through the real SkillPivotPipeline.

    Regression test for round 11 review P2 finding.
    """
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class StatefulEnvWithCrashingReset:
        def __init__(self, fail_with_exception=True):
            self.fail_with_exception = fail_with_exception
            self.actions = []

        def reset(self, attempt=1):
            if self.fail_with_exception:
                raise RuntimeError("checkpoint store offline")
            return False

        def step(self, action: str):
            self.actions.append(action)
            return "Failed in old env", True, 0.0

    class CleanNewEnv:
        def __init__(self):
            self.actions = []

        def step(self, action: str):
            self.actions.append(action)
            return "Item successfully acquired in clean env", True, 1.0

    attempt_counter = 0

    def mock_chat_cross_retry(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            attempt_counter += 1
            return "<think>Execute action</think><action>fetch_item</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed to fetch item",
                "teacher_correction": "Execute fetch_item cleanly",
                "skill_gap": "Need clean fetch rule",
                "evolution_direction": "Add fetch rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Clean Fetch",
                "content": "Fetch item cleanly using updated environment handle.",
                "trigger_condition": "When fetching items",
                "rationale": "Succeeds via clean builder",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.98,
                "generalizable_rule_score": 0.95,
                "critique": "Valid improvement",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_cross_retry)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_cross_retry)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_cross_retry)

    # 场景 1: 旧环境 reset 抛出 RuntimeError，重试由独立 builder 提供全新干净环境
    old_env_exc = StatefulEnvWithCrashingReset(fail_with_exception=True)
    clean_envs_created = []

    def cross_builder_exc(attempt: int = 1):
        if attempt == 1:
            return old_env_exc.step
        new_env = CleanNewEnv()
        clean_envs_created.append(new_env)
        return new_env.step

    pipeline_cross_1 = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline_cross_1.teacher_engine.max_retries = 3

    case_cross_1 = PairedSuffixCase(
        task_id="task_p2_cross_exc",
        task_description="Old env reset raises error, clean builder recovers.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_1, accepted_deltas_1, _ = pipeline_cross_1.optimize_step(
        paired_cases=[case_cross_1],
        current_skill="# Initial Skill",
        env_step_fn=old_env_exc.step,
        env_step_fn_builder=cross_builder_exc,
        adapter=None,
    )

    assert case_cross_1.teacher_success is True
    assert len(accepted_deltas_1) == 1
    assert "Fetch item cleanly using updated environment handle." in updated_skill_1
    assert old_env_exc.actions == ["fetch_item"]
    assert len(clean_envs_created) == 1
    assert clean_envs_created[0].actions == ["fetch_item"]

    # 场景 2: 旧环境 reset 返回 False，重试由独立 builder 提供全新干净环境
    old_env_false = StatefulEnvWithCrashingReset(fail_with_exception=False)
    clean_envs_created_false = []

    def cross_builder_false(attempt: int = 1):
        if attempt == 1:
            return old_env_false.step
        new_env = CleanNewEnv()
        clean_envs_created_false.append(new_env)
        return new_env.step

    pipeline_cross_2 = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline_cross_2.teacher_engine.max_retries = 3

    case_cross_2 = PairedSuffixCase(
        task_id="task_p2_cross_false",
        task_description="Old env reset returns False, clean builder recovers.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_2, accepted_deltas_2, _ = pipeline_cross_2.optimize_step(
        paired_cases=[case_cross_2],
        current_skill="# Initial Skill",
        env_step_fn=old_env_false.step,
        env_step_fn_builder=cross_builder_false,
        adapter=None,
    )

    assert case_cross_2.teacher_success is True
    assert len(accepted_deltas_2) == 1
    assert "Fetch item cleanly using updated environment handle." in updated_skill_2
    assert old_env_false.actions == ["fetch_item"]
    assert len(clean_envs_created_false) == 1
    assert clean_envs_created_false[0].actions == ["fetch_item"]


# ── 19. Round 12 Regression: Explicit Step Function Priority Over Adapter Hook ──


def test_explicit_env_step_fn_priority_over_adapter_default_none_hook(monkeypatch):
    """Verify that when both an adapter (whose build_continuation_step_fn returns None by default)
    and an explicit fixed env_step_fn are provided to SkillPivotPipeline.optimize_step, the explicit
    env_step_fn is reliably executed rather than discarded, preventing unreached terminal states
    from being falsely accepted as stateless QA success.

    Regression test for round 12 review P1 finding:
    Prior code placed the adapter continuation branch before callable(env_step_fn) and directly
    returned adapter.build_continuation_step_fn(case) without falling back when it returned None,
    causing real environment callbacks to be skipped and gold_answers to falsely pass.
    """
    from skillopt.envs.base import EnvAdapter
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    step_calls = []

    def mock_env_step(action: str):
        step_calls.append(action)
        # Even if teacher provides FINISH: 4, the environment explicitly reports non-terminal done=False
        return "goal is not yet reached", False, 0.0

    class DefaultHookAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["default_hook"]

    def mock_chat_priority(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Direct finish</think><action>FINISH: 4</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_priority)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_priority)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_priority)

    adapter = DefaultHookAdapter()
    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=2)

    case = PairedSuffixCase(
        task_id="task_p1_priority",
        task_description="Solve calculation with environment step.",
        ground_truth="4",
        gold_answers=["4"],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill, accepted_deltas, _ = pipeline.optimize_step(
        paired_cases=[case],
        current_skill="# Initial Skill",
        env_step_fn=mock_env_step,
        adapter=adapter,
    )

    # Verification:
    # 1. The explicit step callback MUST have been executed
    assert len(step_calls) > 0
    assert step_calls[0] == "FINISH: 4"
    # 2. Since the environment step returned done=False, the task MUST NOT be accepted as success
    assert case.teacher_success is False
    assert len(accepted_deltas) == 0
    assert updated_skill == "# Initial Skill"


# ── 20. Round 12 Regression: Recreated Closure Reusing Same Polluted Environment ─


def test_recreated_closure_reusing_polluted_underlying_env_rejected_while_fresh_builder_succeeds(monkeypatch):
    """Verify that when an adapter returns a new closure callable on each attempt but the closure
    operates on the same underlying stateful environment without reset, the pipeline accurately
    detects the underlying instance pollution via closure inspection and blocks attempt 2 with RESET_ERROR,
    preventing non-replayable suffixes from being accepted. Also verifies that a genuine fresh builder
    succeeds cleanly.

    Regression test for round 12 review P1 finding.
    """
    from skillopt.envs.base import EnvAdapter
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    class SneakyClosureAdapter(EnvAdapter):
        def __init__(self):
            super().__init__()
            self.door_open = False
            self.step_calls = []

        def build_train_env(self, batch_size, seed, **kwargs):
            return []

        def build_eval_env(self, env_num, split, seed, **kwargs):
            return []

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return []

        def get_task_types(self):
            return ["closure_test"]

        def build_continuation_step_fn(self, case, **kwargs):
            # Every invocation returns a brand new function closure object (new callable identity)
            def closure_step(action: str):
                self.step_calls.append(action)
                if action == "open":
                    self.door_open = True
                    return "door opened", False, 0.0
                elif action == "collect":
                    if self.door_open:
                        return "item collected", True, 1.0
                    return "door closed", True, 0.0
                return "unknown action", False, 0.0

            return closure_step

    attempt_counter = 0

    def mock_chat_closure(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            attempt_counter += 1
            if attempt_counter == 1:
                return "<think>Open door</think><action>open</action>", {}
            else:
                return "<think>Collect item</think><action>collect</action>", {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_closure)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_closure)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_closure)

    # ── Branch A: Sneaky closure reusing same adapter state ──
    adapter = SneakyClosureAdapter()
    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline.teacher_engine.max_retries = 2

    case_sneaky = PairedSuffixCase(
        task_id="task_p1_sneaky_closure",
        task_description="Open door and collect reward.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_sneaky, accepted_deltas_sneaky, _ = pipeline.optimize_step(
        paired_cases=[case_sneaky],
        current_skill="# Initial Skill",
        adapter=adapter,
    )

    # Verification: attempt 2 was blocked by RESET_ERROR because closure_step captured self (polluted adapter)
    assert case_sneaky.teacher_success is False
    assert len(accepted_deltas_sneaky) == 0
    assert updated_skill_sneaky == "# Initial Skill"
    assert adapter.step_calls == ["open"]
    assert not any(step.action == "collect" for step in case_sneaky.teacher_suffix)
    assert len(case_sneaky.teacher_suffix) == 1
    assert case_sneaky.teacher_suffix[0].action == "open"

    # ── Branch B: Genuine clean builder creating fresh instance on each attempt ──
    class FreshDoorEnv:
        def __init__(self):
            self.door_open = False
            self.actions = []

        def step(self, action: str):
            self.actions.append(action)
            if action == "open_and_collect":
                self.door_open = True
                return "door opened and item collected", True, 1.0
            return "failed", True, 0.0

    fresh_envs_created = []

    def fresh_builder(attempt: int = 1):
        env = FreshDoorEnv()
        fresh_envs_created.append(env)
        # Returns fresh closure capturing fresh env
        def fresh_step(action: str):
            return env.step(action)
        return fresh_step

    def mock_chat_fresh(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            attempt_counter += 1
            if attempt_counter == 1:
                return "<think>Wrong action</think><action>wrong</action>", {}
            else:
                return "<think>Correct composite action</think><action>open_and_collect</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed action",
                "teacher_correction": "Use open_and_collect",
                "skill_gap": "Need open_and_collect rule",
                "evolution_direction": "Add open_and_collect rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Door Operation",
                "content": "Execute open_and_collect atomically.",
                "trigger_condition": "When dealing with closed door",
                "rationale": "Succeeds cleanly",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.96,
                "generalizable_rule_score": 0.95,
                "critique": "Solid update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_fresh)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_fresh)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_fresh)

    pipeline_fresh = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline_fresh.teacher_engine.max_retries = 2

    case_fresh = PairedSuffixCase(
        task_id="task_p1_fresh_builder",
        task_description="Open door and collect cleanly.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_fresh, accepted_deltas_fresh, _ = pipeline_fresh.optimize_step(
        paired_cases=[case_fresh],
        current_skill="# Initial Skill",
        env_step_fn_builder=fresh_builder,
        adapter=None,
    )

    assert case_fresh.teacher_success is True
    assert len(accepted_deltas_fresh) == 1
    assert "Execute open_and_collect atomically." in updated_skill_fresh
    assert len(fresh_envs_created) == 2
    assert fresh_envs_created[0].actions == ["wrong"]
    assert fresh_envs_created[1].actions == ["open_and_collect"]


# ── 21. Round 12 Regression: Unhashable Dataclass Environment and Callable ──────


def test_unhashable_dataclass_environment_and_callable_retry_lifecycle(monkeypatch):
    """Verify that mutable dataclasses (with default __hash__=None) and callable dataclasses
    can be used as environments and step functions across successful first attempts and
    multi-attempt retry lifecycles without raising TypeError: unhashable type.

    Regression test for round 12 review P2 finding.
    """
    from dataclasses import dataclass, field
    from skillopt.optimizer.skillpivot import delta_generator, regression_verifier, teacher_continuation
    from skillopt.optimizer.skillpivot.pipeline import SkillPivotPipeline
    from skillopt.types import DeviationPoint, PairedSuffixCase

    def mock_chat_dataclass(system, user, stage="", **kw):
        if stage == "teacher_continuation":
            return "<think>Act</think><action>do_task</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Failed task",
                "teacher_correction": "Do task correctly",
                "skill_gap": "Need task rule",
                "evolution_direction": "Add task rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Task Execution",
                "content": "Follow unhashable dataclass execution.",
                "trigger_condition": "When running dataclass env",
                "rationale": "Succeeds without TypeError",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Solid update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_dataclass)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_dataclass)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_dataclass)

    pipeline = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)

    # ── Branch A: Normal mutable dataclass environment with method binding ──
    @dataclass
    class UnhashableDataEnv:
        actions: list[str] = field(default_factory=list)

        def step(self, action: str):
            self.actions.append(action)
            return "goal reached", True, 1.0

    env_a = UnhashableDataEnv()
    assert getattr(env_a, "__hash__") is None  # explicitly verify unhashable

    case_a = PairedSuffixCase(
        task_id="task_p2_unhashable_env",
        task_description="Execute dataclass step.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill_a, accepted_deltas_a, _ = pipeline.optimize_step(
        paired_cases=[case_a],
        current_skill="# Initial Skill",
        env_step_fn=env_a.step,
        adapter=None,
    )

    assert case_a.teacher_success is True
    assert len(accepted_deltas_a) == 1
    assert "Follow unhashable dataclass execution." in updated_skill_a
    assert env_a.actions == ["do_task"]

    # ── Branch B: Callable mutable dataclass as env_step_fn directly ──
    @dataclass
    class CallableUnhashableStepEnv:
        actions: list[str] = field(default_factory=list)

        def __call__(self, action: str):
            self.actions.append(action)
            return "goal reached", True, 1.0

    callable_env = CallableUnhashableStepEnv()
    assert getattr(callable_env, "__hash__") is None

    case_b = PairedSuffixCase(
        task_id="task_p2_callable_unhashable",
        task_description="Execute callable dataclass step.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    updated_skill_b, accepted_deltas_b, _ = pipeline.optimize_step(
        paired_cases=[case_b],
        current_skill="# Initial Skill",
        env_step_fn=callable_env,
        adapter=None,
    )

    assert case_b.teacher_success is True
    assert len(accepted_deltas_b) == 1
    assert callable_env.actions == ["do_task"]

    # ── Branch C: Mutable dataclass across retry with successful reset ──
    @dataclass
    class StatefulUnhashableResetEnv:
        count: int = 0
        actions: list[str] = field(default_factory=list)

        def reset(self, attempt=1):
            self.count = 0
            return True

        def step(self, action: str):
            self.actions.append(action)
            self.count += 1
            if self.count == 1 and action == "wrong":
                return "failed", True, 0.0
            return "success after reset", True, 1.0

    attempt_counter = 0

    def mock_chat_dataclass_retry(system, user, stage="", **kw):
        nonlocal attempt_counter
        if stage == "teacher_continuation":
            attempt_counter += 1
            if attempt_counter == 1:
                return "<think>Try wrong</think><action>wrong</action>", {}
            else:
                return "<think>Try right</think><action>right</action>", {}
        elif stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Wrong action",
                "teacher_correction": "Right action",
                "skill_gap": "Need right rule",
                "evolution_direction": "Add right rule",
            }), {}
        elif stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Retry Rule",
                "content": "Reset and execute right action.",
                "trigger_condition": "When retrying dataclass env",
                "rationale": "Succeeds on attempt 2",
            }), {}
        elif stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.95,
                "generalizable_rule_score": 0.95,
                "critique": "Solid update",
            }), {}
        return "{}", {}

    monkeypatch.setattr(teacher_continuation, "chat_optimizer", mock_chat_dataclass_retry)
    monkeypatch.setattr(delta_generator, "chat_optimizer", mock_chat_dataclass_retry)
    monkeypatch.setattr(regression_verifier, "chat_optimizer", mock_chat_dataclass_retry)

    env_c = StatefulUnhashableResetEnv()
    pipeline_retry = SkillPivotPipeline(allow_simulated=False, teacher_max_steps=1)
    pipeline_retry.teacher_engine.max_retries = 2

    case_c = PairedSuffixCase(
        task_id="task_p2_unhashable_retry_reset",
        task_description="Execute dataclass retry with reset.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_c, accepted_deltas_c, _ = pipeline_retry.optimize_step(
        paired_cases=[case_c],
        current_skill="# Initial Skill",
        env_step_fn=env_c.step,
        adapter=None,
    )

    assert case_c.teacher_success is True
    assert len(accepted_deltas_c) == 1
    assert "Reset and execute right action." in updated_skill_c
    assert env_c.actions == ["wrong", "right"]

    # ── Branch D: Mutable dataclass across retry without reset (rejected by gate 2) ──
    @dataclass
    class StatefulUnhashableNoResetEnv:
        actions: list[str] = field(default_factory=list)

        def step(self, action: str):
            self.actions.append(action)
            return "failed step", True, 0.0

    env_d = StatefulUnhashableNoResetEnv()
    case_d = PairedSuffixCase(
        task_id="task_p2_unhashable_retry_no_reset",
        task_description="Dataclass without reset should be rejected safely.",
        ground_truth="",
        gold_answers=[],
        prefix_steps=[],
        deviation=DeviationPoint(step_index=1, trough_potential=0.0, pre_mean_potential=0.0, post_mean_potential=0.0),
    )

    attempt_counter = 0
    updated_skill_d, accepted_deltas_d, _ = pipeline_retry.optimize_step(
        paired_cases=[case_d],
        current_skill="# Initial Skill",
        env_step_fn=env_d.step,
        adapter=None,
    )

    assert case_d.teacher_success is False
    assert len(accepted_deltas_d) == 0
    assert updated_skill_d == "# Initial Skill"
    assert env_d.actions == ["wrong"]
    assert len(case_d.teacher_suffix) == 1
    assert case_d.teacher_suffix[0].action == "wrong"








