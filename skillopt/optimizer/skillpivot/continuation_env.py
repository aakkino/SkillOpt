"""Resolve and restore continuation environments once per teacher attempt."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

logger = logging.getLogger(__name__)
StepFunction = Callable[[str], tuple[str, bool, float]]


@dataclass(frozen=True, eq=False)
class ContinuationEnvironment:
    """A step callback and the actual state it mutates.

    ``state`` is the environment object, compared by identity, not equality.
    ``restored`` certifies restoration to the case's prefix for THIS build.
    Return a new handle after each restoration; reusing a consumed handle does
    not certify another restoration. Shared logs/configuration are not state.
    ``reset_fn`` optionally restores this state to the same prefix.
    """

    step_fn: StepFunction
    state: Any
    restored: bool = False
    reset_fn: Callable[..., bool | None] | None = None

    def __post_init__(self):
        if not callable(self.step_fn) or self.state is None:
            raise ValueError("ContinuationEnvironment requires a callable and a state object")

    def __call__(self, action: str) -> tuple[str, bool, float]:
        return self.step_fn(action)


def call_hook(fn: Callable, *args, **context):
    """Bind compatible arguments before calling; never retry a body TypeError."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **context)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        context = {
            k: v
            for k, v in context.items()
            if k in sig.parameters
            and sig.parameters[k].kind in (sig.parameters[k].POSITIONAL_OR_KEYWORD, sig.parameters[k].KEYWORD_ONLY)
        }
    sig.bind(*args, **context)
    return fn(*args, **context)


def call_reset(fn: Callable, attempt: int):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(attempt)
    for args, kwargs in [((attempt,), {}), ((), {"attempt": attempt}), ((), {})]:
        try:
            sig.bind(*args, **kwargs)
        except TypeError:
            continue
        return fn(*args, **kwargs)
    raise TypeError("reset callback must accept an attempt or no arguments")


def _environment_states(step_fn: Any) -> list[Any]:
    """Infer legacy callback state, excluding incidental closure dependencies.

    Bound receivers and objects exposing step/continuation hooks are environment
    candidates. Unlabelled dict/list state needs an explicit environment handle;
    arbitrary captured logs and configuration are never treated as environments.
    """
    states: list[Any] = []
    seen: list[Any] = []
    pending = [(step_fn, True)]

    def append_state(obj):
        if obj is not None and not any(obj is item for item in states):
            states.append(obj)

    while pending:
        obj, root = pending.pop()
        if obj is None or any(obj is item for item in seen):
            continue
        seen.append(obj)
        if isinstance(obj, ContinuationEnvironment):
            append_state(obj.state)
            continue
        if isinstance(obj, partial):
            pending.append((obj.func, True))
            pending.extend((arg, False) for arg in obj.args)
            pending.extend((arg, False) for arg in (obj.keywords or {}).values())
            continue
        receiver = getattr(obj, "__self__", None)
        if receiver is not None:
            # Captured helpers such as trace.append/logger.info are not the
            # callback's environment. Wrapped env.step still exposes step.
            if root or any(callable(getattr(receiver, name, None)) for name in ("step", "build_continuation_step_fn")):
                append_state(receiver)
            pending.append((receiver, False))
        explicit_states = [getattr(obj, name, None) for name in ("env", "environment", "underlying_env", "target_env")]
        for state in explicit_states:
            append_state(state)
        if not inspect.isroutine(obj) and not inspect.isclass(obj):
            if any(callable(getattr(obj, name, None)) for name in ("step", "build_continuation_step_fn")):
                if not any(state is not None for state in explicit_states):
                    append_state(obj)
            elif root and callable(obj):
                append_state(obj)
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is not None:
            pending.append((wrapped, True))
        for cell in getattr(obj, "__closure__", None) or ():
            try:
                pending.append((cell.cell_contents, False))
            except ValueError:
                pass
        pending.extend((arg, False) for arg in getattr(obj, "__defaults__", None) or ())
        pending.extend((arg, False) for arg in (getattr(obj, "__kwdefaults__", None) or {}).values())
    return states


class ContinuationSetupError(RuntimeError):
    def __init__(self, action: str, message: str):
        self.action = action
        super().__init__(message)


class ContinuationRuntime:
    """Own callback selection, restoration, and persistent dirty state tracking."""

    def __init__(self, case, step_fn=None, builder=None, reset_fn=None, adapter=None, train_env=None, context=None):
        self.case = case
        self.step_fn = step_fn
        self.builder = builder
        self.reset_fn = reset_fn
        self.adapter = adapter
        self.context = {**(context or {}), "train_env": train_env}
        self.dirty: list[Any] = []
        self.used_handles: list[ContinuationEnvironment] = []
        self.opaque_used = False
        self.executed_environment = False
        self.active = None
        self.states: list[Any] = []

    def _build(self, attempt):
        if self.builder is not None:
            return self.builder(attempt)
        if self.step_fn is not None:
            # Compatibility for the original pipeline's case/attempt factories.
            try:
                params = list(inspect.signature(self.step_fn).parameters)
            except (TypeError, ValueError):
                params = []
            if params and params[0] in ("case", "task_case", "paired_case"):
                if len(params) > 1 and params[1] in ("attempt", "retry"):
                    return self.step_fn(self.case, attempt)
                return self.step_fn(self.case)
            if params and params[0] in ("attempt", "retry"):
                return self.step_fn(attempt)
            return self.step_fn
        hook = getattr(self.adapter, "build_continuation_step_fn", None)
        if hook is not None:
            return call_hook(hook, self.case, **{**self.context, "attempt": attempt})
        return None

    def _adapter_reset(self):
        from skillopt.envs.base import EnvAdapter

        hook = getattr(self.adapter, "reset_continuation_env", None)
        if hook is None or getattr(hook, "__func__", None) is EnvAdapter.reset_continuation_env:
            return None
        return hook

    def _restore(self, attempt):
        resets: list[tuple[str, Callable]] = []
        if isinstance(self.active, ContinuationEnvironment) and self.active.reset_fn is not None:
            resets.append(("environment reset", self.active.reset_fn))
        if self.reset_fn is not None:
            resets.append(("env_reset_fn", self.reset_fn))
        adapter_reset = self._adapter_reset()
        if adapter_reset is not None:
            resets.append(
                (
                    "adapter.reset_continuation_env",
                    lambda: call_hook(adapter_reset, self.case, **{**self.context, "attempt": attempt}),
                )
            )
        reset_owners: list[Any] = []
        for obj in [self.active, *self.states]:
            if any(obj is owner for owner in reset_owners):
                continue
            reset_owners.append(obj)
            reset = getattr(obj, "reset", None)
            if callable(reset):
                resets.append(("active_env_step_fn reset", reset))
        errors = []
        for label, reset in resets:
            try:
                if call_reset(reset, attempt) is False:
                    raise RuntimeError(f"{label} returned False")
                self.dirty = [
                    ref for ref in self.dirty if not any(ref is state for state in (self.states or [self.active]))
                ]
                return
            except Exception as exc:
                logger.warning("Task %s: %s error on attempt %s: %s", self.case.task_id, label, attempt, exc)
                errors.append(str(exc))
        reason = "; ".join(errors) or "stateful environment not reset from previous attempt"
        raise ContinuationSetupError("RESET_ERROR", f"Environment reset failed: {reason}")

    def prepare(self, attempt):
        try:
            self.active = self._build(attempt)
            if self.active is not None and not callable(self.active):
                raise TypeError("continuation builder must return a callable, environment handle, or None")
            if self.active is None and self.executed_environment:
                raise RuntimeError("interactive continuation cannot switch to answer-only verification")
        except Exception as exc:
            label = (
                "env_step_fn_builder error"
                if self.builder is not None or self.step_fn is not None
                else "failed to rebuild step fn"
            )
            logger.warning("Task %s: %s on attempt %s: %s", self.case.task_id, label, attempt, exc)
            raise ContinuationSetupError("ENVIRONMENT_ERROR", f"Environment construction failed: {exc}") from exc
        self.states = _environment_states(self.active)
        if self.active is None:
            return None
        # Opaque callbacks may hide dictionary/global state. A changed wrapper
        # is not proof of recovery; builders can return an explicit handle.
        identities = self.states or [self.active]
        dirty = any(state is old for state in identities for old in self.dirty)
        dirty = dirty or (not self.states and self.opaque_used)
        certified = (
            isinstance(self.active, ContinuationEnvironment)
            and self.active.restored is True
            and not any(self.active is old for old in self.used_handles)
        )
        if certified:
            self.dirty = [old for old in self.dirty if not any(old is state for state in identities)]
        elif dirty:
            self._restore(attempt)
        return self.active

    def mark_used(self):
        if self.active is None:
            return
        self.executed_environment = True
        self.opaque_used = self.opaque_used or not self.states
        for state in self.states or [self.active]:
            if not any(state is old for old in self.dirty):
                self.dirty.append(state)
        if isinstance(self.active, ContinuationEnvironment):
            if not any(self.active is old for old in self.used_handles):
                self.used_handles.append(self.active)
