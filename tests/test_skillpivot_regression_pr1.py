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
