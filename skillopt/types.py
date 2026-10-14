"""Standardized I/O types for the ReflACT pipeline.

Shared dataclass definitions for the 6-stage per-step pipeline
and the 2 epoch-level stages.  All types support round-trip
conversion to/from plain dicts for incremental adoption.

Re-exports
----------
GateResult, GateAction — from skillopt.evaluation.gate
BatchSpec              — from skillopt.datasets.base
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
import re
from typing import Any, Literal

from skillopt.datasets.base import BatchSpec  # noqa: F401
from skillopt.evaluation.gate import GateAction, GateResult  # noqa: F401

# ── Atomic types ─────────────────────────────────────────────────────────

EditOp = Literal["append", "insert_after", "replace", "delete"]


@dataclass
class Edit:
    """A single edit operation on a skill document.

    Used across Reflect → Aggregate → Select → Update → MetaReflect.
    """

    op: EditOp
    content: str = ""
    target: str = ""
    support_count: int | None = None
    source_type: Literal["failure", "success"] | None = None
    merge_level: int | None = None
    update_origin: str = ""
    update_target: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> Edit:
        return cls(
            op=d.get("op", "append"),
            content=d.get("content", ""),
            target=d.get("target", ""),
            support_count=d.get("support_count"),
            source_type=d.get("source_type"),
            merge_level=d.get("merge_level"),
            update_origin=d.get("update_origin", ""),
            update_target=d.get("update_target", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"op": self.op, "content": self.content}
        if self.target:
            d["target"] = self.target
        if self.support_count is not None:
            d["support_count"] = self.support_count
        if self.source_type is not None:
            d["source_type"] = self.source_type
        if self.merge_level is not None:
            d["merge_level"] = self.merge_level
        if self.update_origin:
            d["update_origin"] = self.update_origin
        if self.update_target:
            d["update_target"] = self.update_target
        return d


@dataclass
class Patch:
    """A set of edits with reasoning.

    Output of Aggregate (③), Select (④); input to Update (⑤).
    """

    edits: list[Edit] = field(default_factory=list)
    reasoning: str = ""
    ranking_details: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Patch:
        edits_raw = d.get("edits", [])
        return cls(
            edits=[Edit.from_dict(e) if isinstance(e, dict) else e for e in edits_raw],
            reasoning=d.get("reasoning", ""),
            ranking_details=d.get("ranking_details"),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "edits": [e.to_dict() if isinstance(e, Edit) else e for e in self.edits],
        }
        if self.ranking_details is not None:
            d["ranking_details"] = self.ranking_details
        return d


# ── Stage ① ROLLOUT ──────────────────────────────────────────────────────

@dataclass
class RolloutResult:
    """Result of a single episode/task rollout.

    Universal fields are required; env-specific fields live in ``extras``.
    """

    id: str
    hard: float
    soft: float
    n_turns: int = 0
    fail_reason: str = ""
    task_type: str = ""
    task_description: str = ""
    predicted_answer: str = ""
    question: str = ""
    reference_text: str = ""
    target_system_prompt: str = ""
    target_user_prompt: str = ""
    spreadsheet_preview: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    _KNOWN_FIELDS: frozenset[str] | None = field(
        default=None, init=False, repr=False, compare=False,  # type: ignore[assignment]
    )

    @classmethod
    def _get_known_fields(cls) -> frozenset[str]:
        if cls._KNOWN_FIELDS is None:
            cls._KNOWN_FIELDS = frozenset(
                f.name for f in dc_fields(cls)
                if f.name != "_KNOWN_FIELDS"
            )
        return cls._KNOWN_FIELDS

    @classmethod
    def from_dict(cls, d: dict) -> RolloutResult:
        known = cls._get_known_fields()
        extras = {k: v for k, v in d.items() if k not in known}
        return cls(
            id=str(d.get("id", "")),
            hard=float(d.get("hard", 0)),
            soft=float(d.get("soft", 0.0)),
            n_turns=int(d.get("n_turns", 0)),
            fail_reason=str(d.get("fail_reason", "")),
            task_type=str(d.get("task_type", "")),
            task_description=str(d.get("task_description", "")),
            predicted_answer=str(d.get("predicted_answer", "")),
            question=str(d.get("question", "")),
            reference_text=str(d.get("reference_text", "")),
            target_system_prompt=str(d.get("target_system_prompt", "")),
            target_user_prompt=str(d.get("target_user_prompt", "")),
            spreadsheet_preview=str(d.get("spreadsheet_preview", "")),
            extras=extras,
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "hard": self.hard,
            "soft": self.soft,
        }
        for attr in (
            "n_turns", "fail_reason", "task_type", "task_description",
            "predicted_answer", "question", "reference_text",
            "target_system_prompt", "target_user_prompt",
            "spreadsheet_preview",
        ):
            val = getattr(self, attr)
            if val:
                d[attr] = val
        d.update(self.extras)
        return d


# ── Stage ② REFLECT ──────────────────────────────────────────────────────

@dataclass
class FailureSummaryEntry:
    """One entry in the failure summary produced by error analysts."""

    failure_type: str
    count: int = 0
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> FailureSummaryEntry:
        return cls(
            failure_type=d.get("failure_type", ""),
            count=int(d.get("count", 0)),
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "failure_type": self.failure_type,
            "count": self.count,
            "description": self.description,
        }


@dataclass
class RawPatch:
    """Analyst output from the Reflect stage — a patch with provenance.

    Wraps the dict produced by ``run_error_analyst_minibatch``
    and ``run_success_analyst_minibatch``.
    """

    patch: Patch
    source_type: Literal["failure", "success"] = "failure"
    batch_size: int = 0
    failure_summary: list[FailureSummaryEntry] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> RawPatch | None:
        if d is None:
            return None
        inner = d.get("patch", d)
        if not isinstance(inner, dict):
            return None
        patch = Patch.from_dict(inner)
        return cls(
            patch=patch,
            source_type=d.get("source_type", "failure"),
            batch_size=int(d.get("batch_size", 0)),
            failure_summary=[
                FailureSummaryEntry.from_dict(fs)
                for fs in d.get("failure_summary", [])
            ],
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "patch": self.patch.to_dict(),
            "source_type": self.source_type,
            "batch_size": self.batch_size,
        }
        if self.failure_summary:
            d["failure_summary"] = [fs.to_dict() for fs in self.failure_summary]
        return d


# ── Epoch-level: SLOW_UPDATE ─────────────────────────────────────────────

@dataclass
class SlowUpdateResult:
    """Output of the epoch-level slow update stage (EMA / regularization)."""

    reasoning: str = ""
    slow_update_content: str = ""
    action: str = ""
    time_s: float | None = None
    prev_hard: float | None = None
    curr_hard: float | None = None
    selection_hard: float | None = None
    selection_soft: float | None = None
    candidate_hash: str = ""
    update_origin: str = ""
    update_target: str = ""

    @classmethod
    def from_dict(cls, d: dict | None) -> SlowUpdateResult | None:
        if d is None:
            return None
        return cls(
            reasoning=d.get("reasoning", ""),
            slow_update_content=d.get("slow_update_content", ""),
            action=d.get("action", ""),
            time_s=d.get("time_s"),
            prev_hard=d.get("prev_hard"),
            curr_hard=d.get("curr_hard"),
            selection_hard=d.get("selection_hard"),
            selection_soft=d.get("selection_soft"),
            candidate_hash=d.get("candidate_hash", ""),
            update_origin=d.get("update_origin", ""),
            update_target=d.get("update_target", ""),
        )

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "reasoning": self.reasoning,
            "slow_update_content": self.slow_update_content,
        }
        if self.action:
            d["action"] = self.action
        if self.time_s is not None:
            d["time_s"] = self.time_s
        if self.prev_hard is not None:
            d["prev_hard"] = self.prev_hard
        if self.curr_hard is not None:
            d["curr_hard"] = self.curr_hard
        if self.selection_hard is not None:
            d["selection_hard"] = self.selection_hard
        if self.selection_soft is not None:
            d["selection_soft"] = self.selection_soft
        if self.candidate_hash:
            d["candidate_hash"] = self.candidate_hash
        if self.update_origin:
            d["update_origin"] = self.update_origin
        if self.update_target:
            d["update_target"] = self.update_target
        return d


# ── SkillPivot Extension: Fine-grained Slices & Deviation Types ──────────

@dataclass
class TrajectoryStep:
    """Standardized single-step interaction node.

    Compatible with conversation.json records across environments (tool-call
    or standard ReAct step records).
    """

    step: int
    action: str = ""
    observation: str = ""
    reasoning: str = ""
    raw_model_response: str = ""
    reward: float = 0.0
    done: bool = False
    execution_status: int = 1  # 1: success/usable, 0: syntax error/exception/empty (Exec_t)

    @staticmethod
    def is_system_or_eval(d: dict) -> bool:
        """Check whether a conversation record is system metadata or evaluation report."""
        if not isinstance(d, dict):
            return False
        role = str(d.get("role", "")).lower()
        if role in {"system", "evaluator"}:
            return True
        msg_type = str(d.get("type", "")).lower()
        if msg_type in {"system", "eval", "evaluation", "eval_result"}:
            return True
        content = str(d.get("content", ""))
        if "[EVALUATION RESULT]" in content or content.strip().startswith("Exact Match:"):
            return True
        return False

    @classmethod
    def from_dict(cls, d: dict, default_step: int = 0) -> TrajectoryStep:
        # Tool-call record: {"type": "tool_call", "cmd": ..., "obs": ...}
        if d.get("type") == "tool_call" or "cmd" in d:
            obs = str(d.get("obs") or "")
            action = str(d.get("cmd") or "")
            is_valid = 1 if action and obs and "error" not in obs.lower() else 0
            return cls(
                step=int(d.get("step", default_step)),
                action=action,
                observation=obs,
                reasoning=str(d.get("reasoning") or ""),
                execution_status=is_valid,
                done=bool(d.get("done", False)),
            )

        # Message record (e.g. SearchQA: {"type": "message", "turn": 1, "content": ...})
        if d.get("type") == "message" or ("content" in d and "action" not in d):
            content = str(d.get("content") or "")
            # Extract reasoning (<think>...</think>)
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL | re.IGNORECASE)
            reasoning = think_match.group(1).strip() if think_match else ""

            # Extract action / answer
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)

            if action_match:
                action = action_match.group(1).strip()
            elif answer_match:
                action = f"<answer>{answer_match.group(1).strip()}</answer>"
            else:
                clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
                action = clean_content if clean_content else content.strip()

            obs = str(d.get("env_feedback") or d.get("observation") or d.get("obs") or "")
            if not obs and answer_match:
                obs = "(final answer generated)"

            step_idx = int(d.get("step") or d.get("turn") or default_step)
            is_valid = 1 if action and "error" not in action.lower() else 0

            return cls(
                step=step_idx,
                action=action,
                observation=obs,
                reasoning=reasoning,
                raw_model_response=content,
                reward=float(d.get("reward", 0.0)),
                done=bool(d.get("done", bool(answer_match))),
                execution_status=is_valid,
            )

        # Standard Step record: {"step": N, "action": ..., "env_feedback": ...}
        action = str(d.get("action") or "")
        obs = str(d.get("env_feedback") or d.get("observation") or "")
        exec_st = int(d.get("execution_status", 1))
        if not action or "error" in obs.lower() or obs.strip() == "":
            exec_st = 0

        return cls(
            step=int(d.get("step", default_step)),
            action=action,
            observation=obs,
            reasoning=str(d.get("reasoning") or ""),
            raw_model_response=str(d.get("model_response") or ""),
            reward=float(d.get("reward", 0.0)),
            done=bool(d.get("done", False)),
            execution_status=exec_st,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "observation": self.observation,
            "reasoning": self.reasoning,
            "raw_model_response": self.raw_model_response,
            "reward": self.reward,
            "done": self.done,
            "execution_status": self.execution_status,
        }


@dataclass
class StepPotential:
    """Step progress potential and individual factor scores."""

    step: int
    exec_score: float      # Exec_t: tool execution validity
    prog_score: float      # Prog_t: goal relevance positive increase
    div_score: float       # Div_t: action diversity distance
    potential: float       # V_t = w1*Exec_t + w2*Prog_t + w3*Div_t
    is_trough: bool = False


@dataclass
class DeviationPoint:
    """Deviation point detection decision object."""

    step_index: int                    # Deviation truncation point t* (1-indexed)
    trough_potential: float            # Potential score at t*
    pre_mean_potential: float          # Average potential before t* (1 to t*-1)
    post_mean_potential: float         # Average potential from t* onwards
    is_confirmed: bool = False         # Confirmed by sustained low potential
    reason: str = ""                   # Diagnostic rationale
    diagnostics: list[StepPotential] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "trough_potential": self.trough_potential,
            "pre_mean_potential": self.pre_mean_potential,
            "post_mean_potential": self.post_mean_potential,
            "is_confirmed": self.is_confirmed,
            "reason": self.reason,
            "diagnostics": [
                {
                    "step": p.step,
                    "exec_score": p.exec_score,
                    "prog_score": p.prog_score,
                    "div_score": p.div_score,
                    "potential": p.potential,
                    "is_trough": p.is_trough,
                }
                for p in self.diagnostics
            ],
        }


@dataclass
class PairedSuffixCase:
    """SkillPivot core paired slice container.

    Holds: prefix_steps (1 to t*-1) + student_suffix (t* to T) + teacher_suffix (t* to t*+K).
    """

    task_id: str
    task_type: str = ""
    task_description: str = ""
    ground_truth: str = ""             # Ground-truth for relevance and verification
    skill_name: str = ""
    skill_content: str = ""            # Active skill snapshot used during rollout

    deviation: DeviationPoint | None = None

    # Three-stage structured slices
    prefix_steps: list[TrajectoryStep] = field(default_factory=list)      # Steps 1 to t*-1
    student_suffix: list[TrajectoryStep] = field(default_factory=list)    # Steps t* to T
    teacher_suffix: list[TrajectoryStep] = field(default_factory=list)    # Teacher continuation (max 10 steps)

    teacher_success: bool = False
    contrast_summary: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def slice_from_full_trajectory(
        cls,
        task_id: str,
        full_steps: list[TrajectoryStep],
        deviation: DeviationPoint,
        *,
        task_type: str = "",
        task_description: str = "",
        ground_truth: str = "",
        skill_name: str = "",
        skill_content: str = "",
    ) -> PairedSuffixCase:
        """Partition full student trajectory into prefix and student_suffix at t*."""
        t_star = deviation.step_index
        prefix = [s for s in full_steps if s.step < t_star]
        student_sfx = [s for s in full_steps if s.step >= t_star]
        return cls(
            task_id=task_id,
            task_type=task_type,
            task_description=task_description,
            ground_truth=ground_truth,
            skill_name=skill_name,
            skill_content=skill_content,
            deviation=deviation,
            prefix_steps=prefix,
            student_suffix=student_sfx,
            teacher_suffix=[],
            teacher_success=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "ground_truth": self.ground_truth,
            "skill_name": self.skill_name,
            "deviation": self.deviation.to_dict() if self.deviation else None,
            "prefix_steps": [s.to_dict() for s in self.prefix_steps],
            "student_suffix": [s.to_dict() for s in self.student_suffix],
            "teacher_suffix": [s.to_dict() for s in self.teacher_suffix],
            "teacher_success": self.teacher_success,
            "contrast_summary": self.contrast_summary,
            "extras": self.extras,
        }


SkillDeltaAction = Literal["improve_skill", "optimize_description", "skip"]


@dataclass
class SkillDelta:
    """SkillPivot minimal atomic delta patch."""

    action: SkillDeltaAction
    skill_name: str = ""
    target_rule: str = ""
    content: str = ""
    trigger_condition: str = ""
    rationale: str = ""
    source_case_id: str = ""

    def render_content(self) -> str:
        """Render candidate SkillDelta to textual representation preserving procedural update header,
        applicable condition, and rule content."""
        rule_suffix = f": {self.target_rule.strip()}" if self.target_rule.strip() else ""
        header = f"### Procedural Update [{self.action}]{rule_suffix}\n"
        if self.trigger_condition.strip():
            header += f"> Applicable Condition: {self.trigger_condition.strip()}\n\n"
        return (header + self.content.strip()).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "skill_name": self.skill_name,
            "target_rule": self.target_rule,
            "content": self.content,
            "trigger_condition": self.trigger_condition,
            "rationale": self.rationale,
            "source_case_id": self.source_case_id,
        }


@dataclass
class RegressionGateResult:
    """Prior semantic review gate result."""

    no_regression_score: float         # in [0, 1], required >= 0.5
    generalizable_rule_score: float    # in [0, 1], required >= 0.5
    threshold: float = 0.5
    is_accepted: bool = False
    critique: str = ""

    def __post_init__(self):
        self.is_accepted = (
            self.no_regression_score >= self.threshold
            and self.generalizable_rule_score >= self.threshold
        )

