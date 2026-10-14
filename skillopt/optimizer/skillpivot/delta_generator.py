"""Paired Suffix Contrast and Skill Delta Generator for SkillPivot.

Implements Stage 4 of SkillPivot (arXiv:2609.29154):
  - Contrasts student's failed suffix against teacher's successful suffix
    under the identical shared prefix.
  - Phase 1: Single-case Trajectory Summarizer: analyzes student mistake,
    teacher correction, skill gap, and evolution direction.
  - Phase 2: Skill Increment Generator: produces minimal atomic skill delta
    with action in {"improve_skill", "optimize_description", "skip"}.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from skillopt.model import chat_optimizer
from skillopt.types import PairedSuffixCase, SkillDelta, TrajectoryStep


# ── Prompt Templates ─────────────────────────────────────────────────────

CONTRAST_ANALYSIS_SYSTEM = """You are an expert diagnostic analyst for AI Agent Skills.
Your role is to contrast a student agent's failed trajectory suffix against a teacher agent's successful suffix starting from the EXACT SAME trajectory prefix.
Identify why the student failed, how the teacher succeeded, and diagnose the missing procedural knowledge in the active skill.

Return your analysis in valid JSON format with keys:
{
  "student_mistake": "specific deviation or error in student suffix",
  "teacher_correction": "how teacher solved the problem from the same point",
  "skill_gap": "unclear, missing, or misleading procedural guidance in the skill",
  "evolution_direction": "abstract reusable procedural rule to add or refine"
}"""

DELTA_GENERATION_SYSTEM = """You are a Skill Increment Generator.
Based on the contrastive analysis of a failure and its successful teacher recovery, propose a MINIMAL, AUDITABLE, and CONDITIONAL skill delta.

Choose exactly one action:
- "improve_skill": add a localized procedural recovery rule or operational protocol.
- "optimize_description": refine trigger conditions, boundaries, or applicability.
- "skip": do not modify if the evidence is insufficient or instance-specific.

Output valid JSON only:
{
  "action": "improve_skill" | "optimize_description" | "skip",
  "target_rule": "section or rule in the skill to modify/augment",
  "content": "precise, compact markdown guidance to insert or update",
  "trigger_condition": "when this procedural guidance applies",
  "rationale": "why this minimal delta repairs the observed failure"
}"""


# ── Helpers ──────────────────────────────────────────────────────────────


def _format_steps(steps: list[TrajectoryStep]) -> str:
    if not steps:
        return "(empty)\n"
    lines = []
    for s in steps:
        lines.append(f"Step {s.step}:")
        if s.reasoning:
            lines.append(f"  Think: {s.reasoning}")
        lines.append(f"  Action: {s.action}")
        lines.append(f"  Obs: {s.observation[:300]}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any]:
    """Robustly extract JSON dictionary from markdown codeblocks or raw text."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw_json = match.group(1)
    else:
        # Try finding outermost braces
        start = text.find("{")
        end = text.rfind("}")
        raw_json = text[start : end + 1] if start != -1 and end != -1 else text

    try:
        return json.loads(raw_json)
    except Exception:
        return {}


# ── Paired Suffix Contrast Generator ─────────────────────────────────────


class DeltaGenerator:
    """Generates minimal SkillDelta through paired suffix contrast."""

    def __init__(
        self,
        chat_fn: Callable[..., tuple[str, dict]] = chat_optimizer,
        max_completion_tokens: int = 4096,
    ):
        self.chat_fn = chat_fn
        self.max_completion_tokens = max_completion_tokens

    def summarize_contrast(self, case: PairedSuffixCase) -> dict[str, str]:
        """Phase 1: Summarize behavioral difference between student and teacher."""
        user_prompt = (
            f"## Task Description\n{case.task_description}\n\n"
            f"## Active Skill\n{case.skill_content.strip() or '(none)'}\n\n"
            f"## Shared Prefix (Steps 1 to {len(case.prefix_steps)})\n"
            f"{_format_steps(case.prefix_steps)}\n\n"
            f"## Student's Failed Suffix\n"
            f"{_format_steps(case.student_suffix)}\n\n"
            f"## Teacher's Successful Suffix\n"
            f"{_format_steps(case.teacher_suffix)}\n\n"
            "Analyze the contrast and output JSON as instructed."
        )

        response, _ = self.chat_fn(
            system=CONTRAST_ANALYSIS_SYSTEM,
            user=user_prompt,
            max_completion_tokens=self.max_completion_tokens,
            stage="contrast_analysis",
        )

        parsed = _extract_json(response)
        case.contrast_summary = json.dumps(parsed, ensure_ascii=False)
        return parsed

    def generate_delta(
        self,
        case: PairedSuffixCase,
        contrast_summary: dict[str, str] | None = None,
    ) -> SkillDelta:
        """Phase 2: Generate candidate SkillDelta from contrast summary."""
        if contrast_summary is None:
            contrast_summary = self.summarize_contrast(case)

        user_prompt = (
            f"## Active Skill Content\n{case.skill_content}\n\n"
            f"## Contrastive Analysis Summary\n"
            f"- Student Mistake: {contrast_summary.get('student_mistake', '')}\n"
            f"- Teacher Correction: {contrast_summary.get('teacher_correction', '')}\n"
            f"- Identified Skill Gap: {contrast_summary.get('skill_gap', '')}\n"
            f"- General Direction: {contrast_summary.get('evolution_direction', '')}\n\n"
            "Generate the minimal, auditable SkillDelta JSON."
        )

        response, _ = self.chat_fn(
            system=DELTA_GENERATION_SYSTEM,
            user=user_prompt,
            max_completion_tokens=self.max_completion_tokens,
            stage="delta_generation",
        )

        parsed = _extract_json(response)
        raw_action = parsed.get("action", "improve_skill").strip().lower()
        action = raw_action if raw_action in {"improve_skill", "optimize_description", "skip"} else "improve_skill"

        return SkillDelta(
            action=action,
            skill_name=case.skill_name,
            target_rule=parsed.get("target_rule", ""),
            content=parsed.get("content", ""),
            trigger_condition=parsed.get("trigger_condition", ""),
            rationale=parsed.get("rationale", ""),
            source_case_id=case.task_id,
        )

    @staticmethod
    def apply_delta(original_skill: str, delta: SkillDelta) -> str:
        """Apply candidate SkillDelta to original skill to produce tentative skill s'."""
        if delta.action == "skip" or not delta.content.strip():
            return original_skill

        header = f"\n\n### Procedural Update [{delta.action}]: {delta.target_rule}\n"
        if delta.trigger_condition.strip():
            header += f"> Applicable Condition: {delta.trigger_condition.strip()}\n\n"

        return original_skill.strip() + header + delta.content.strip() + "\n"
