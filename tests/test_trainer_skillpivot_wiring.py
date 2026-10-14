"""Tests for trainer SkillPivot routing and wiring."""
import json
import os
import pytest

from skillopt.config import flatten_config, load_config
from skillopt.engine.trainer import run_skillpivot_reflect_step
from skillopt.types import TrajectoryStep


def test_skillpivot_config_resolution():
    """Verify loading and flattening of skillpivot configuration."""
    cfg = flatten_config(load_config("configs/searchqa/skillpivot.yaml"))
    assert cfg.get("optimizer_type") == "skillpivot"
    assert cfg.get("skillpivot_embedding_backend") == "local"
    assert cfg.get("skillpivot_gate_threshold") == 0.5
    assert cfg.get("skillpivot_teacher_max_steps") == 10


def test_run_skillpivot_reflect_step_no_failures():
    """Verify that all-success rollout results immediately return empty patches."""
    rollout_results = [
        {"id": "env_001", "hard": 1.0, "soft": 1.0},
        {"id": "env_002", "hard": 1.0, "soft": 1.0},
    ]
    patches = run_skillpivot_reflect_step(
        rollout_results=rollout_results,
        current_skill="# Skill",
        prediction_dir="dummy_dir",
        cfg={"optimizer_type": "skillpivot"},
    )
    assert patches == []


def test_run_skillpivot_reflect_step_with_mocked_pipeline(tmp_path, monkeypatch):
    """Verify the full wiring from conversation.json to accepted raw_patches."""
    pred_dir = tmp_path / "predictions"
    task_dir = pred_dir / "task_fail_01"
    task_dir.mkdir(parents=True)

    # Prepare conversation.json
    conversation = [
        {"step": 1, "action": "search keyword", "env_feedback": "relevant docs", "execution_status": 1},
        {"step": 2, "action": "loop click", "env_feedback": "404 not found", "execution_status": 0},
        {"step": 3, "action": "repeat loop click", "env_feedback": "404 not found", "execution_status": 0},
    ]
    with open(task_dir / "conversation.json", "w", encoding="utf-8") as f:
        json.dump(conversation, f)

    rollout_results = [
        {
            "id": "task_fail_01",
            "hard": 0.0,
            "soft": 0.0,
            "task_description": "Search financial numbers",
            "reference_text": "Net income: $20M",
        }
    ]

    # Mock chat_optimizer to avoid any real network call
    def mock_chat_optimizer(system, user, stage="", **kwargs):
        if stage == "teacher_continuation":
            return "<think>Direct query</think><action>FINISH: $20M</action>", {}
        if stage == "contrast_analysis":
            return json.dumps({
                "student_mistake": "Repeated 404",
                "teacher_correction": "Direct query",
                "skill_gap": "Missing retry rule",
                "evolution_direction": "Add retry fallback",
            }), {}
        if stage == "delta_generation":
            return json.dumps({
                "action": "improve_skill",
                "target_rule": "Fallback Guidelines",
                "content": "On 404, fallback to synonyms.",
                "trigger_condition": "404 error",
                "rationale": "Prevents dead loops",
            }), {}
        if stage == "regression_verification":
            return json.dumps({
                "no_regression_score": 0.9,
                "generalizable_rule_score": 0.9,
                "critique": "Valid general rule",
            }), {}
        return "", {}

    monkeypatch.setattr("skillopt.optimizer.skillpivot.teacher_continuation.chat_optimizer", mock_chat_optimizer)
    monkeypatch.setattr("skillopt.optimizer.skillpivot.delta_generator.chat_optimizer", mock_chat_optimizer)
    monkeypatch.setattr("skillopt.optimizer.skillpivot.regression_verifier.chat_optimizer", mock_chat_optimizer)

    cfg = {
        "optimizer_type": "skillpivot",
        "skillpivot_embedding_backend": "local",
        "skillpivot_gate_threshold": 0.5,
    }

    from tests.test_deviation_detector import MockEmbeddingProvider

    raw_patches = run_skillpivot_reflect_step(
        rollout_results=rollout_results,
        current_skill="# Search Skill\nAlways verify keywords.",
        prediction_dir=str(pred_dir),
        cfg=cfg,
        embedding_provider=MockEmbeddingProvider(),
    )

    assert len(raw_patches) == 1
    patch_item = raw_patches[0]
    assert patch_item["source_type"] == "failure"
    inner_patch = patch_item["patch"]
    assert "SkillPivot [improve_skill]" in inner_patch["reasoning"]
    assert len(inner_patch["edits"]) == 1
    edit = inner_patch["edits"][0]
    assert edit["op"] == "append"
    assert edit["target"] == "Fallback Guidelines"
    assert "On 404, fallback to synonyms." in edit["content"]
