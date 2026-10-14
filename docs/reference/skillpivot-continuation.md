# SkillPivot continuation environments

Teacher retries must start from the same saved student prefix. Environment
selection and recovery are owned by `ContinuationRuntime`; Pipeline forwards
the adapter, training context, and explicit callbacks to the teacher engine.

Selection priority is an explicit `env_step_fn_builder(attempt)`, an explicit
`env_step_fn` (including the existing case/attempt factory convention), then
`adapter.build_continuation_step_fn(case, train_env=..., attempt=...)`.
Explicit evaluators also take priority over adapter hooks. Adapter evaluators
are resolved per case, with the real training context.

## Explicit state handles

Adapters can still return a step callable or `None` for answer-only QA. For
wrapped callbacks, dictionary state, or callbacks sharing adapter/logging
objects, return `ContinuationEnvironment`:

```python
from skillopt.optimizer.skillpivot import ContinuationEnvironment

def build_continuation_step_fn(self, case, train_env=None, attempt=1):
    env = self.restore_from_prefix(case, train_env)

    def step(action):
        self.audit_log.append((case.task_id, attempt, action))
        return env.step(action)

    return ContinuationEnvironment(step_fn=step, state=env, restored=True)
```

`state` must identify the actual mutable environment, using the same object
when callbacks share that state. It need not be hashable. Shared logs and
configuration do not belong in this field. A new state object represents an
independent environment; the adapter must initialize it from the saved prefix.

`restored=True` certifies that this build successfully restored the state to
the prefix. It is a per-build result, not a permanent “fresh” attribute. Return
a new handle after each successful restoration. Reusing a handle that already
executed actions cannot reuse its restoration certificate. Legacy `is_fresh`,
`is_clean_env`, and `checkpoint_restored` flags are not restoration proofs.

A handle may also supply `reset_fn(attempt)` (or a zero-argument reset) to
restore its own state. Adapter `reset_continuation_env`, explicit
`env_reset_fn`, and legacy environment `.reset` callbacks remain supported.
`None`/`True` means successful restoration; `False` or an exception means
failure. A successful recovery removes pollution only for the current state.
Failed/skipped attempts retain all previously mutated state records.

## Legacy callbacks and failure behavior

Bound methods, `functools.partial`, decorators, default arguments, and closures
capturing step-capable environments remain supported. Captured logging,
configuration, and ordinary lists/dictionaries are not inferred as task state.
Opaque legacy callbacks can run initially, but retries require a reset or an
explicit handle; creating another wrapper alone does not prove recovery.

Construction failures reject the attempt before a model call. A dirty state
with no successful recovery is rejected before further actions. A newly built
independent state remains usable even when the old state cannot reset. Tool
execution exceptions cannot be converted into success by matching an answer.
After an interactive attempt executes, a builder cannot silently switch the
same case to answer-only verification by returning `None`.

Hook arguments are bound before invocation. A `TypeError` inside a hook/reset
is an implementation failure, not a reason to invoke the same operation again
with fewer arguments. Unsuccessful cases never reach delta generation.
