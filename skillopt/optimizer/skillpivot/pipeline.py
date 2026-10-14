"""SkillPivot Unified Optimization Pipeline.

Coordinates the end-to-end 5-stage SkillPivot workflow:
  1. Student Rollout (handled by trainer / adapter)
  2. Deviation Detection (slices trajectories into PairedSuffixCase)
  3. Teacher Continuation (takes over from t* to succeed)
  4. Paired Suffix Contrast (extracts minimal SkillDelta)
  5. Regression-Gated Verification (accepts if No-Regression >= 0.5 & Generalizable-Rule >= 0.5)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from skillopt.optimizer.skillpivot.delta_generator import DeltaGenerator
from skillopt.optimizer.skillpivot.regression_verifier import RegressionGatedVerifier
from skillopt.optimizer.skillpivot.teacher_continuation import TeacherContinuationEngine
from skillopt.types import PairedSuffixCase, RegressionGateResult, SkillDelta

logger = logging.getLogger(__name__)


class SkillPivotPipeline:
    """Orchestrates Teacher Continuation, Contrastive Delta Generation, and Semantic Gating."""

    def __init__(
        self,
        teacher_engine: TeacherContinuationEngine | None = None,
        delta_generator: DeltaGenerator | None = None,
        verifier: RegressionGatedVerifier | None = None,
        gate_threshold: float = 0.5,
        allow_simulated: bool = False,
        success_reward_threshold: float = 0.0,
        teacher_max_steps: int = 10,
    ):
        self.allow_simulated = allow_simulated
        self.teacher_engine = teacher_engine or TeacherContinuationEngine(
            allow_simulated=allow_simulated,
            success_reward_threshold=success_reward_threshold,
            max_steps=teacher_max_steps,
        )
        self.delta_generator = delta_generator or DeltaGenerator()
        self.verifier = verifier or RegressionGatedVerifier(threshold=gate_threshold)

    def optimize_step(
        self,
        paired_cases: list[PairedSuffixCase],
        current_skill: str,
        env_step_fn: Callable[[str], tuple[str, bool, float]] | None = None,
        evaluator_fn: Callable[[str, Any], bool] | None = None,
        allow_simulated: bool | None = None,
        adapter: Any = None,
        train_env: Any = None,
        **kwargs,
    ) -> tuple[str, list[SkillDelta], list[RegressionGateResult]]:
        """Run one optimization iteration across a batch of failed cases.

        Returns:
            updated_skill: new skill document (or unchanged if all rejected)
            accepted_deltas: list of accepted SkillDelta objects
            gate_results: audit records for each candidate delta
        """
        updated_skill = current_skill
        accepted_deltas: list[SkillDelta] = []
        gate_results: list[RegressionGateResult] = []
        sim_allowed = self.allow_simulated if allow_simulated is None else allow_simulated

        for case in paired_cases:
            last_reset_attempt = [0]

            def _env_reset(attempt: int = 1):
                if attempt > 1 and last_reset_attempt[0] != attempt:
                    if adapter is not None and hasattr(adapter, "reset_continuation_env"):
                        try:
                            adapter.reset_continuation_env(case, train_env=train_env, attempt=attempt, **kwargs)
                        except TypeError:
                            try:
                                adapter.reset_continuation_env(case, train_env=train_env, attempt=attempt)
                            except TypeError:
                                try:
                                    adapter.reset_continuation_env(case, train_env=train_env)
                                except TypeError:
                                    adapter.reset_continuation_env(case)
                    last_reset_attempt[0] = attempt

            def _step_fn_builder(attempt: int = 1):
                if attempt > 1:
                    try:
                        _env_reset(attempt)
                    except Exception as e:
                        logger.warning(f"Task {case.task_id}: failed to reset env on attempt {attempt}: {e}")
                if callable(env_step_fn):
                    try:
                        import inspect
                        sig = inspect.signature(env_step_fn)
                        params = list(sig.parameters.keys())
                        if len(params) >= 2 and params[0] in ("case", "task_case", "paired_case") and params[1] in ("attempt", "retry"):
                            return env_step_fn(case, attempt)
                        elif len(params) >= 1 and params[0] in ("case", "task_case", "paired_case"):
                            return env_step_fn(case)
                        elif len(params) >= 1 and params[0] in ("attempt", "retry"):
                            return env_step_fn(attempt)
                        else:
                            return env_step_fn
                    except Exception:
                        return env_step_fn
                elif adapter is not None and hasattr(adapter, "build_continuation_step_fn"):
                    try:
                        return adapter.build_continuation_step_fn(
                            case,
                            train_env=train_env,
                            attempt=attempt,
                            **kwargs,
                        )
                    except TypeError:
                        try:
                            return adapter.build_continuation_step_fn(case, train_env=train_env)
                        except TypeError:
                            return adapter.build_continuation_step_fn(case)
                    except Exception as e:
                        logger.warning(f"Task {case.task_id}: failed to build continuation step fn from adapter: {e}")
                        return None
                return None

            # Resolve task-specific evaluator function
            case_evaluator_fn = evaluator_fn
            if adapter is not None and hasattr(adapter, "build_continuation_evaluator_fn"):
                try:
                    adapter_eval = adapter.build_continuation_evaluator_fn(
                        case,
                        train_env=train_env,
                        **kwargs,
                    )
                    if adapter_eval is not None:
                        case_evaluator_fn = adapter_eval
                except TypeError:
                    try:
                        adapter_eval = adapter.build_continuation_evaluator_fn(case)
                        if adapter_eval is not None:
                            case_evaluator_fn = adapter_eval
                    except Exception:
                        pass
                except Exception:
                    pass

            # Stage 3: Teacher Continuation (if not already simulated/run)
            if not case.teacher_success or not case.teacher_suffix:
                case = self.teacher_engine.run_continuation(
                    case,
                    env_step_fn=_step_fn_builder(1),
                    evaluator_fn=case_evaluator_fn,
                    allow_simulated=sim_allowed,
                    adapter=adapter,
                    train_env=train_env,
                    env_step_fn_builder=_step_fn_builder,
                    env_reset_fn=_env_reset,
                )

            if not case.teacher_success:
                logger.info(f"Task {case.task_id}: Teacher continuation failed. Skipping case.")
                continue

            # Stage 4: Paired Suffix Contrast & Delta Generation
            delta = self.delta_generator.generate_delta(case)
            if delta.action == "skip" or not delta.content.strip():
                logger.info(f"Task {case.task_id}: Delta generator chose SKIP.")
                continue

            # Tentative application
            tentative_skill = self.delta_generator.apply_delta(updated_skill, delta)

            # Stage 5: Regression-Gated Review
            gate_res = self.verifier.verify_update(
                original_skill=updated_skill,
                tentative_skill=tentative_skill,
                delta=delta,
            )
            gate_results.append(gate_res)

            if gate_res.is_accepted:
                logger.info(
                    f"Task {case.task_id}: Skill delta ACCEPTED "
                    f"(No-Reg={gate_res.no_regression_score:.2f}, Gen-Rule={gate_res.generalizable_rule_score:.2f})"
                )
                updated_skill = tentative_skill
                accepted_deltas.append(delta)
            else:
                logger.warning(
                    f"Task {case.task_id}: Skill delta REJECTED by gate "
                    f"(No-Reg={gate_res.no_regression_score:.2f}, Gen-Rule={gate_res.generalizable_rule_score:.2f})"
                )

        return updated_skill, accepted_deltas, gate_results
