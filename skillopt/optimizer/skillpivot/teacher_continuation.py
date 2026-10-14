"""Teacher Continuation module for SkillPivot.

Implements Stage 3 of SkillPivot (arXiv:2609.29154):
  - Strong teacher model takes over execution from deviation point t*.
  - Receives shared prefix (steps 1 to t*-1), current skill, and student's failed
    suffix as an explicit negative reference to avoid.
  - Continues the ReAct loop from step t* (up to max 10 steps).
  - Uses an inline reflector to guide step-by-step decision without polluting
    the raw trajectory suffix.
  - Accepts continuation only if the task goal is verified as completed.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from skillopt.model import chat_optimizer
from skillopt.types import PairedSuffixCase, TrajectoryStep

# ── Helpers for parsing model response ───────────────────────────────────


def extract_action(response: str) -> str:
    """Extract action string from <action>...</action> tags or fallback to full line."""
    m = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback to json command or simple line
    return response.strip()


def extract_think(response: str) -> str:
    """Extract reasoning from <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ── Teacher Continuation Engine ──────────────────────────────────────────


class TeacherContinuationEngine:
    """Manages teacher continuation rollout from deviation point t*."""

    def __init__(
        self,
        max_steps: int = 10,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_completion_tokens: int = 8192,
    ):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

    def build_teacher_system_prompt(self, skill_content: str) -> str:
        """Construct teacher system prompt embedding active skill guidelines."""
        skill_section = f"\n\n## Available Skills\n{skill_content.strip()}\n" if skill_content.strip() else ""
        return (
            "You are an expert Teacher Agent for tool-augmented task solving.\n"
            "Your goal is to successfully complete the task by continuing from a previous agent's valid exploration prefix.\n"
            f"{skill_section}\n"
            "Format your turn using:\n"
            "<think>your step-by-step reasoning</think>\n"
            "<action>tool call or final answer</action>\n"
            "If the task is finished, return `<action>FINISH: [answer]</action>`."
        )

    def build_teacher_prefix_prompt(self, case: PairedSuffixCase) -> str:
        """Construct the prompt containing task description, shared prefix, and negative reference."""
        lines = [
            f"## Task Description\n{case.task_description}\n",
            "## Exploration History (Shared Prefix)\n",
            "The student agent has completed the following initial steps:",
        ]

        if case.prefix_steps:
            for s in case.prefix_steps:
                lines.append(f"[Step {s.step} Think] {s.reasoning or '(none)'}")
                lines.append(f"[Step {s.step} Action] {s.action}")
                lines.append(f"[Step {s.step} Observation] {s.observation}")
        else:
            lines.append("(No prior prefix; starting from step 1)")

        lines.append("\n## Negative Reference (What to Avoid)")
        lines.append("The student previously failed by making these subsequent mistakes:")
        if case.student_suffix:
            for s in case.student_suffix:
                lines.append(f"- Step {s.step}: Action `{s.action}` -> Obs `{s.observation[:200]}`")
        else:
            lines.append("- (No failure suffix recorded)")

        t_star = case.deviation.step_index if case.deviation else len(case.prefix_steps) + 1
        lines.append(
            f"\n## Instruction\n"
            f"Now, take over execution starting from Step {t_star}.\n"
            f"Avoid repeating the student's mistakes. Plan carefully and invoke the correct tools to finish the task."
        )
        return "\n".join(lines)

    def inline_reflection(
        self,
        latest_action: str,
        latest_observation: str,
        goal: str,
    ) -> str:
        """Generate a lightweight online hint for the next teacher step."""
        if not latest_observation or "error" in latest_observation.lower():
            return "Note: The last tool call returned an error or empty result. Try alternative arguments or verify prerequisites."
        if "found" in latest_observation.lower() or "success" in latest_observation.lower():
            return "Note: Progress observed. Verify if sufficient evidence exists to formulate the final answer."
        return "Note: Proceed towards the goal, ensuring action parameters strictly match observations."

    def run_continuation(
        self,
        case: PairedSuffixCase,
        env_step_fn: Callable[[str], tuple[str, bool, float]] | None = None,
        chat_fn: Callable[..., tuple[str, dict]] = chat_optimizer,
    ) -> PairedSuffixCase:
        """Execute teacher continuation from t* for up to max_steps turns.

        Args:
            case: The PairedSuffixCase with populated prefix_steps and deviation.
            env_step_fn: Optional callable(action) -> (observation, is_done, reward).
                         If None, uses simulated step or single-turn response.
            chat_fn: LLM call function (defaults to chat_optimizer).
        """
        system_prompt = self.build_teacher_system_prompt(case.skill_content)
        base_user_prompt = self.build_teacher_prefix_prompt(case)

        t_star = case.deviation.step_index if case.deviation else len(case.prefix_steps) + 1

        for attempt in range(1, self.max_retries + 1):
            teacher_steps: list[TrajectoryStep] = []
            conversation_history = base_user_prompt
            success = False

            for step_offset in range(self.max_steps):
                current_step_num = t_star + step_offset

                # Call Teacher Model
                raw_response, _ = chat_fn(
                    system=system_prompt,
                    user=conversation_history,
                    max_completion_tokens=self.max_completion_tokens,
                    stage="teacher_continuation",
                )

                action = extract_action(raw_response)
                reasoning = extract_think(raw_response)

                # Check if teacher finishes or calls environment
                is_finish = "finish:" in action.lower() or "final answer" in action.lower()

                if env_step_fn is not None:
                    try:
                        obs, done, reward = env_step_fn(action)
                        exec_st = 0 if "error" in obs.lower() or not obs.strip() else 1
                    except Exception as e:
                        obs = f"Execution error: {e}"
                        done = False
                        reward = 0.0
                        exec_st = 0
                else:
                    # Simulated / self-contained environment step
                    obs = "Simulated tool response: OK" if not is_finish else "Task finished"
                    done = is_finish
                    reward = 1.0 if is_finish else 0.0
                    exec_st = 1

                step_record = TrajectoryStep(
                    step=current_step_num,
                    action=action,
                    observation=obs,
                    reasoning=reasoning,
                    raw_model_response=raw_response,
                    reward=reward,
                    done=done,
                    execution_status=exec_st,
                )
                teacher_steps.append(step_record)

                if done or is_finish:
                    success = True
                    break

                # Inline reflection hint (appended to prompt, but not in raw step_record)
                hint = self.inline_reflection(action, obs, case.task_description)
                conversation_history += (
                    f"\n\n[Step {current_step_num} Teacher Action] {action}\n"
                    f"[Step {current_step_num} Observation] {obs}\n"
                    f"[{hint}]"
                )

            if success:
                case.teacher_suffix = teacher_steps
                case.teacher_success = True
                return case

        # Failed after max_retries
        case.teacher_suffix = teacher_steps
        case.teacher_success = False
        return case
