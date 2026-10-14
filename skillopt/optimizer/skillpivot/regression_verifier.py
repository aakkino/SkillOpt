"""Regression-Gated Verifier for SkillPivot.

Implements Stage 5 of SkillPivot (arXiv:2609.29154):
  - Prior semantic gate evaluated before expensive validation rollouts.
  - Reviews original skill s vs tentative updated skill s' under two criteria:
    1. No-Regression: ensures s' preserves useful rules from s and does not
       delete, weaken, or contradict them without justification.
    2. Generalizable-Rule: checks whether newly added guidance expresses a
       reusable procedural principle rather than an instance-specific patch.
  - Updates accepted if and only if both scores >= theta (default theta = 0.5).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from skillopt.model import chat_optimizer
from skillopt.types import RegressionGateResult, SkillDelta

# ── Reviewer System Prompt ───────────────────────────────────────────────

VERIFIER_SYSTEM_PROMPT = """You are a rigorous Regression-Gated Skill Reviewer for LLM Agent systems.
Evaluate a proposed skill update s' against the original skill s using two independent criteria:

1. "no_regression" (score between 0.0 and 1.0):
   - Does s' preserve useful and valid rules from s?
   - Does s' avoid deleting, weakening, or directly contradicting existing guidelines?
   - A score >= 0.5 means existing stable capabilities are safely preserved.

2. "generalizable_rule" (score between 0.0 and 1.0):
   - Does the added guidance express a general, reusable procedural workflow (e.g. error recovery, formatting, fallback)?
   - Does it avoid overfitting to instance-specific entities, constant answers, or single task details?
   - A score >= 0.5 means the rule generalizes beyond the triggering case.

Output valid JSON only:
{
  "no_regression_score": <float between 0.0 and 1.0>,
  "generalizable_rule_score": <float between 0.0 and 1.0>,
  "critique": "<detailed rationale for scores>"
}"""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        raw_json = match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        raw_json = text[start : end + 1] if start != -1 and end != -1 else text

    try:
        return json.loads(raw_json)
    except Exception:
        return {}


class RegressionGatedVerifier:
    """Semantic verifier assessing candidate skill updates against regression and overfitting."""

    def __init__(
        self,
        threshold: float = 0.5,
        chat_fn: Callable[..., tuple[str, dict]] | None = None,
        max_completion_tokens: int = 4096,
    ):
        self.threshold = threshold
        self._chat_fn = chat_fn
        self.max_completion_tokens = max_completion_tokens

    @property
    def chat_fn(self) -> Callable[..., tuple[str, dict]]:
        return self._chat_fn or chat_optimizer

    @chat_fn.setter
    def chat_fn(self, fn: Callable[..., tuple[str, dict]] | None) -> None:
        self._chat_fn = fn

    def verify_update(
        self,
        original_skill: str,
        tentative_skill: str,
        delta: SkillDelta | None = None,
    ) -> RegressionGateResult:
        """Evaluate tentative skill s' against original skill s."""
        # Fast path: if delta is skipped or tentative skill identical, reject update safely
        if tentative_skill.strip() == original_skill.strip():
            return RegressionGateResult(
                no_regression_score=1.0,
                generalizable_rule_score=0.0,
                threshold=self.threshold,
                is_accepted=False,
                critique="Tentative skill is identical to original; no update to merge.",
            )

        delta_info = ""
        if delta is not None:
            delta_info = (
                f"\n## Proposed Delta Action: {delta.action}\n"
                f"Target Rule: {delta.target_rule}\n"
                f"Proposed Content:\n{delta.content}\n"
                f"Delta Rationale: {delta.rationale}\n"
            )

        user_prompt = (
            f"## Original Skill (s)\n{original_skill.strip() or '(empty)'}\n\n"
            f"## Tentative Updated Skill (s')\n{tentative_skill.strip()}\n"
            f"{delta_info}\n"
            "Review the changes and return the verification JSON."
        )

        response, _ = self.chat_fn(
            system=VERIFIER_SYSTEM_PROMPT,
            user=user_prompt,
            max_completion_tokens=self.max_completion_tokens,
            stage="regression_verification",
        )

        parsed = _extract_json(response)
        try:
            no_reg = float(parsed.get("no_regression_score", 0.0))
        except (ValueError, TypeError):
            no_reg = 0.0

        try:
            gen_rule = float(parsed.get("generalizable_rule_score", 0.0))
        except (ValueError, TypeError):
            gen_rule = 0.0

        critique = str(parsed.get("critique") or "")

        return RegressionGateResult(
            no_regression_score=no_reg,
            generalizable_rule_score=gen_rule,
            threshold=self.threshold,
            critique=critique,
        )
