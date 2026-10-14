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
        env_step_fn_builder: Callable[[int], Any] | None = None,
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
            from skillopt.envs.base import EnvAdapter

            last_reset_attempt = [0]

            def _env_reset(attempt: int = 1) -> bool:
                if attempt <= 1:
                    return False
                if last_reset_attempt[0] == attempt:
                    return True

                performed = False
                reset_error = None

                # 1. 尝试调用 adapter 的 reset_continuation_env（若非基类空实现）
                if adapter is not None and hasattr(adapter, "reset_continuation_env"):
                    is_base_noop = getattr(type(adapter), "reset_continuation_env", None) is EnvAdapter.reset_continuation_env
                    if not is_base_noop:
                        try:
                            res = adapter.reset_continuation_env(case, train_env=train_env, attempt=attempt, **kwargs)
                            if res is not False:
                                performed = True
                            else:
                                reset_error = "adapter.reset_continuation_env returned False"
                        except TypeError:
                            try:
                                res = adapter.reset_continuation_env(case, train_env=train_env, attempt=attempt)
                                if res is not False:
                                    performed = True
                                else:
                                    reset_error = "adapter.reset_continuation_env returned False"
                            except TypeError:
                                try:
                                    res = adapter.reset_continuation_env(case, train_env=train_env)
                                    if res is not False:
                                        performed = True
                                    else:
                                        reset_error = "adapter.reset_continuation_env returned False"
                                except TypeError:
                                    try:
                                        res = adapter.reset_continuation_env(case)
                                        if res is not False:
                                            performed = True
                                        else:
                                            reset_error = "adapter.reset_continuation_env returned False"
                                    except Exception as e:
                                        reset_error = str(e)
                                        logger.warning(f"Task {case.task_id}: adapter.reset_continuation_env error on attempt {attempt}: {e}")
                                        raise
                            except Exception as e:
                                reset_error = str(e)
                                logger.warning(f"Task {case.task_id}: adapter.reset_continuation_env error on attempt {attempt}: {e}")
                                raise
                        except Exception as e:
                            reset_error = str(e)
                            logger.warning(f"Task {case.task_id}: adapter.reset_continuation_env error on attempt {attempt}: {e}")
                            raise

                # 2. 若 adapter 未重置，尝试调用 env_step_fn 绑定的 reset
                if not performed and callable(env_step_fn):
                    target_reset = None
                    if hasattr(env_step_fn, "reset") and callable(env_step_fn.reset):
                        target_reset = env_step_fn.reset
                    elif hasattr(getattr(env_step_fn, "__self__", None), "reset") and callable(getattr(env_step_fn, "__self__").reset):
                        target_reset = getattr(env_step_fn, "__self__").reset

                    if target_reset is not None:
                        try:
                            res = target_reset(attempt)
                            if res is not False:
                                performed = True
                            else:
                                reset_error = "bound reset returned False"
                        except TypeError:
                            try:
                                res = target_reset()
                                if res is not False:
                                    performed = True
                                else:
                                    reset_error = "bound reset returned False"
                            except Exception as e:
                                reset_error = str(e)
                                logger.warning(f"Task {case.task_id}: env_step_fn bound reset error on attempt {attempt}: {e}")
                                raise
                        except Exception as e:
                            reset_error = str(e)
                            logger.warning(f"Task {case.task_id}: env_step_fn bound reset error on attempt {attempt}: {e}")
                            raise

                if performed:
                    last_reset_attempt[0] = attempt
                    return True
                if reset_error:
                    raise RuntimeError(f"Reset failed on attempt {attempt}: {reset_error}")
                return False

            def _step_fn_builder(attempt: int = 1):
                explicit_builder = env_step_fn_builder or kwargs.get("env_step_fn_builder")
                if explicit_builder is not None:
                    return explicit_builder(attempt)

                # 检查 env_step_fn 是否为可调用的构造工厂函数
                is_factory = False
                factory_call_args = None
                if callable(env_step_fn):
                    try:
                        import inspect
                        sig = inspect.signature(env_step_fn)
                        params = list(sig.parameters.keys())
                        if len(params) >= 2 and params[0] in ("case", "task_case", "paired_case") and params[1] in ("attempt", "retry"):
                            is_factory = True
                            factory_call_args = (case, attempt)
                        elif len(params) >= 1 and params[0] in ("case", "task_case", "paired_case"):
                            is_factory = True
                            factory_call_args = (case,)
                        elif len(params) >= 1 and params[0] in ("attempt", "retry"):
                            is_factory = True
                            factory_call_args = (attempt,)
                    except (ValueError, TypeError):
                        is_factory = False

                if is_factory:
                    # 工厂函数抛出的业务异常直接向外抛出供引擎门控捕获，严禁 fallback 返回工厂本身
                    return env_step_fn(*factory_call_args)

                # 保留显式 step 的优先级，避免 adapter 默认 hook 遮蔽真实 step 执行
                if callable(env_step_fn):
                    if attempt > 1:
                        # 对于纯复用同一固定 env_step_fn 的状态化环境（既无独立 builder 亦非工厂），重试前若存在 reset 机制则必须成功重置
                        # 若重置抛出异常或返回 False，将抛出 RuntimeError 触发构造/重置门控拦截，严禁沿用被污染环境
                        has_reset_mechanism = False
                        if adapter is not None and hasattr(adapter, "reset_continuation_env"):
                            if getattr(type(adapter), "reset_continuation_env", None) is not EnvAdapter.reset_continuation_env:
                                has_reset_mechanism = True
                        if (hasattr(env_step_fn, "reset") and callable(env_step_fn.reset)) or \
                           (hasattr(getattr(env_step_fn, "__self__", None), "reset") and callable(getattr(env_step_fn, "__self__").reset)):
                            has_reset_mechanism = True

                        if has_reset_mechanism:
                            reset_res = _env_reset(attempt)
                            if not reset_res:
                                raise RuntimeError(f"Environment reset failed on attempt {attempt}")
                    return env_step_fn

                # 未显式提供 env_step_fn 时，由 adapter.build_continuation_step_fn 构建
                if adapter is not None and hasattr(adapter, "build_continuation_step_fn"):
                    try:
                        step_cand = adapter.build_continuation_step_fn(
                            case,
                            train_env=train_env,
                            attempt=attempt,
                            **kwargs,
                        )
                        if step_cand is not None:
                            return step_cand
                    except TypeError:
                        try:
                            step_cand = adapter.build_continuation_step_fn(case, train_env=train_env)
                            if step_cand is not None:
                                return step_cand
                        except TypeError:
                            step_cand = adapter.build_continuation_step_fn(case)
                            if step_cand is not None:
                                return step_cand
                    # 保留真实环境构造异常向外抛出，供 TeacherContinuationEngine 捕获并记录失败，严禁吞掉降级为纯答案验证

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

            # 仅在实际存在可用 reset 实现时向引擎传递 reset 回调
            has_adapter_reset = (
                adapter is not None
                and hasattr(adapter, "reset_continuation_env")
                and getattr(type(adapter), "reset_continuation_env", None) is not EnvAdapter.reset_continuation_env
            )
            has_step_reset = (
                callable(env_step_fn)
                and (
                    (hasattr(env_step_fn, "reset") and callable(env_step_fn.reset))
                    or (hasattr(getattr(env_step_fn, "__self__", None), "reset") and callable(getattr(env_step_fn, "__self__").reset))
                )
            )
            effective_env_reset_fn = _env_reset if (has_adapter_reset or has_step_reset) else None

            # Stage 3: Teacher Continuation (if not already simulated/run)
            if not case.teacher_success or not case.teacher_suffix:
                initial_step_fn = env_step_fn if callable(env_step_fn) else None
                case = self.teacher_engine.run_continuation(
                    case,
                    env_step_fn=initial_step_fn,
                    evaluator_fn=case_evaluator_fn,
                    allow_simulated=sim_allowed,
                    adapter=adapter,
                    train_env=train_env,
                    env_step_fn_builder=_step_fn_builder,
                    env_reset_fn=effective_env_reset_fn,
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
