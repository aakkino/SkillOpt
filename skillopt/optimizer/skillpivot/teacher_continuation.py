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
    return response.strip()


def extract_think(response: str) -> str:
    """Extract reasoning from <think>...</think> tags."""
    m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_final_answer(text: str) -> str:
    """Extract candidate final answer from <answer> tags or FINISH: action."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    m_fin = re.search(r"finish:\s*(.*?)(?:</action>|\n|$)", text, re.DOTALL | re.IGNORECASE)
    if m_fin:
        return m_fin.group(1).strip()
    return ""


def default_verify_answer(pred_answer: str, ground_truth: Any) -> bool:
    """Verify if prediction matches ground truth using strict SQuAD-normalized exact matching.

    Rejects empty, punctuation-only, and loose substring or subset matches (e.g. '40' vs '4').
    Supports single or multiple gold answers, key-value prefixes ('Answer: 4'), and full-token
    exact phrase matches in sentences ('$5M' in 'revenue is $5M').
    """
    if not pred_answer or ground_truth is None:
        return False

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    norm_pred = _norm(pred_answer)
    if not norm_pred:
        return False

    if isinstance(ground_truth, str):
        gts = [ground_truth]
    elif isinstance(ground_truth, (list, tuple, set)):
        gts = [str(x) for x in ground_truth]
    else:
        gts = [str(ground_truth)]

    for raw_gt in gts:
        norm_gt = _norm(raw_gt)
        if not norm_gt:
            continue
        # 1. Direct exact match
        if norm_pred == norm_gt:
            return True
        # 2. Key-value prefix format like "Answer: 4" or "Net income: $20M"
        if ":" in raw_gt:
            tail_norm = _norm(raw_gt.split(":", 1)[1])
            if tail_norm and norm_pred == tail_norm:
                return True
        # 3. Contiguous token sequence match within a descriptive sentence
        # (e.g. pred='5m' in gt='revenue is 5m', but NEVER partial digits like '40' in '4')
        pred_toks = norm_pred.split()
        gt_toks = norm_gt.split()
        if pred_toks and len(pred_toks) <= len(gt_toks):
            for i in range(len(gt_toks) - len(pred_toks) + 1):
                if gt_toks[i : i + len(pred_toks)] == pred_toks:
                    return True
    return False


# ── Teacher Continuation Engine ──────────────────────────────────────────


class TeacherContinuationEngine:
    """Manages teacher continuation rollout from deviation point t*."""

    def __init__(
        self,
        max_steps: int = 10,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_completion_tokens: int = 8192,
        allow_simulated: bool = False,
        success_reward_threshold: float = 0.0,
    ):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.allow_simulated = allow_simulated
        self.success_reward_threshold = success_reward_threshold

    def build_teacher_system_prompt(self, skill_content: str, task_type: str = "") -> str:
        """Construct teacher system prompt embedding active skill guidelines and task constraints."""
        skill_section = f"\n\n## Available Skills\n{skill_content.strip()}\n" if skill_content.strip() else ""
        if str(task_type).strip().lower() in ("qa", "single_turn_qa"):
            return (
                "You are an expert Teacher Agent for question-answering task solving.\n"
                "Your goal is to successfully answer the question by reasoning over the provided context passages.\n"
                f"{skill_section}\n"
                "Format your turn using:\n"
                "<think>your step-by-step reasoning</think>\n"
                "<action>FINISH: [answer]</action>\n"
                "Extract and formulate the exact concise answer from the context without unnecessary conversational words."
            )
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
        """Construct prompt containing student-visible task context, description, shared prefix, and negative reference."""
        lines = []

        # Retain student-visible context/evidence without leaking gold answers
        if case.task_context and "## context" not in case.task_description.lower():
            lines.append(f"## Task Context (Evidence)\n{case.task_context.strip()}\n")

        if case.task_description:
            if "## task description" in case.task_description.lower() or "## context" in case.task_description.lower():
                lines.append(f"{case.task_description.strip()}\n")
            else:
                lines.append(f"## Task Description\n{case.task_description.strip()}\n")
        else:
            lines.append("## Task Description\n(No task description provided)\n")

        lines.append("## Exploration History (Shared Prefix)\n")
        lines.append("The student agent has completed the following initial steps:")

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
        evaluator_fn: Callable[[str, Any], bool] | None = None,
        chat_fn: Callable[..., tuple[str, dict]] | None = None,
        allow_simulated: bool | None = None,
    ) -> PairedSuffixCase:
        """Execute teacher continuation from t* for up to max_steps turns.

        Args:
            case: The PairedSuffixCase with populated prefix_steps and deviation.
            env_step_fn: Optional callable(action) -> (observation, is_done, reward).
            evaluator_fn: Optional callable(predicted_answer, ground_truth) -> bool.
            chat_fn: LLM call function (defaults to chat_optimizer).
            allow_simulated: Explicit flag whether unverified simulated execution is accepted.
        """
        call_fn = chat_fn or chat_optimizer
        task_type = getattr(case, "task_type", "") or case.extras.get("task_type", "")
        system_prompt = self.build_teacher_system_prompt(case.skill_content, task_type=task_type)
        base_user_prompt = self.build_teacher_prefix_prompt(case)
        sim_allowed = self.allow_simulated if allow_simulated is None else allow_simulated

        target_gts = (
            case.gold_answers
            or case.extras.get("gold_answers")
            or ([case.ground_truth] if case.ground_truth else [])
        )
        has_verifiable_target = bool(target_gts) or evaluator_fn is not None

        t_star = case.deviation.step_index if case.deviation else len(case.prefix_steps) + 1

        for attempt in range(1, self.max_retries + 1):
            teacher_steps: list[TrajectoryStep] = []
            conversation_history = base_user_prompt
            success = False

            for step_offset in range(self.max_steps):
                current_step_num = t_star + step_offset

                # Call Teacher Model
                raw_response, _ = call_fn(
                    system=system_prompt,
                    user=conversation_history,
                    max_completion_tokens=self.max_completion_tokens,
                    stage="teacher_continuation",
                )

                action = extract_action(raw_response)
                reasoning = extract_think(raw_response)

                # Check if teacher finishes or calls environment
                is_finish = (
                    "finish:" in action.lower()
                    or "final answer" in action.lower()
                    or "<answer>" in raw_response.lower()
                )

                step_success = False

                if env_step_fn is not None:
                    try:
                        obs, done, reward = env_step_fn(action)
                        exec_st = 0 if "error" in obs.lower() or not obs.strip() else 1
                    except Exception as e:
                        obs = f"Execution error: {e}"
                        done = True
                        reward = 0.0
                        exec_st = 0

                    # Respect environment terminal status; do not override done=False on teacher FINISH
                    cand_ans = extract_final_answer(action) or extract_final_answer(raw_response) or obs
                    has_explicit_ans = bool(extract_final_answer(action) or extract_final_answer(raw_response))

                    # Only verify task success when episode is genuinely done
                    if done:
                        if reward > self.success_reward_threshold:
                            # Environment explicitly reported terminal success
                            if evaluator_fn is not None and target_gts and has_explicit_ans:
                                step_success = bool(evaluator_fn(cand_ans, target_gts))
                            elif target_gts and has_explicit_ans:
                                step_success = bool(default_verify_answer(cand_ans, target_gts))
                            else:
                                step_success = True
                        else:
                            # Environment did not give positive reward; verify via explicit evaluator or gold answer
                            if evaluator_fn is not None and target_gts:
                                step_success = bool(evaluator_fn(cand_ans, target_gts))
                            elif target_gts:
                                step_success = bool(default_verify_answer(cand_ans, target_gts))
                            else:
                                step_success = False
                    else:
                        # Intermediate steps with progress/shaping reward (done=False) do NOT count as task success
                        step_success = False
                elif has_verifiable_target:
                    # Verifiable through ground_truth / evaluator without env_step_fn
                    if is_finish:
                        cand_ans = extract_final_answer(action) or extract_final_answer(raw_response)
                        if evaluator_fn is not None:
                            step_success = bool(evaluator_fn(cand_ans, target_gts))
                        else:
                            step_success = bool(default_verify_answer(cand_ans, target_gts))

                        done = True
                        reward = 1.0 if step_success else 0.0
                        obs = "Answer verified: OK" if step_success else "Answer verification failed: incorrect"
                        exec_st = 1 if step_success else 0
                    else:
                        obs = (
                            "Simulated tool response: OK"
                            if sim_allowed
                            else "No environment step function available. Tool interaction is unsupported in this environment; please provide the final answer directly via <action>FINISH: [answer]</action>."
                        )
                        # Do NOT terminate on non-finish actions in verifiable QA mode; allow multi-turn correction up to max_steps
                        done = False
                        reward = 0.0
                        exec_st = 1 if sim_allowed else 0
                        step_success = False
                else:
                    # Neither env_step_fn nor verifiable ground_truth available
                    if sim_allowed:
                        obs = "Simulated tool response: OK" if not is_finish else "Task finished"
                        done = is_finish
                        reward = 1.0 if is_finish else 0.0
                        exec_st = 1
                        step_success = is_finish
                    else:
                        obs = "Error: Unverifiable task outcome (missing env_step_fn and ground_truth)"
                        done = True
                        reward = 0.0
                        exec_st = 0
                        step_success = False

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

                if step_success:
                    success = True
                    break
                if done:
                    # Episode ended without success
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
