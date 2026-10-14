"""SearchQA environment adapter for ReflACT."""
from __future__ import annotations

import json
from typing import Any, Callable

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.searchqa.dataloader import SearchQADataLoader
from skillopt.envs.searchqa.rollout import run_batch
from skillopt.model import get_target_backend


class SearchQAAdapter(EnvAdapter):
    """SearchQA environment adapter."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 120,
        workers: int = 64,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 16384,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = SearchQADataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,  # actually list[dict] for SearchQA
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """Run QA agent on items. Resume-aware."""
        items: list[dict] = env_manager  # type alias for clarity
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
            task_timeout=self.exec_timeout,
        )

    def get_task_types(self) -> list[str]:
        return ["qa"]

    def build_continuation_step_fn(
        self,
        case: Any,
        **kwargs,
    ) -> Callable[[str], tuple[str, bool, float]] | None:
        """SearchQA is a single-turn question-answering benchmark without an interactive tool environment.

        Continuation operates in QA takeover mode verified against gold answers with strict Exact Match.
        """
        return None

    def build_continuation_evaluator_fn(
        self,
        case: Any,
        **kwargs,
    ) -> Callable[[str, Any], bool] | None:
        """Return SearchQA strict EM evaluator for teacher continuation answer verification."""
        from skillopt.envs.searchqa.evaluator import evaluate, normalize_answer

        def _searchqa_eval(pred_ans: str, gold_answers: Any) -> bool:
            norm = normalize_answer(pred_ans)
            if not norm:
                return False
            if isinstance(gold_answers, (list, tuple, set)):
                gold_list = [str(x) for x in gold_answers]
            else:
                gold_list = [str(gold_answers)]
            expanded_gold = list(gold_list)
            for g in gold_list:
                if ":" in g:
                    tail = g.split(":", 1)[1].strip()
                    if tail and tail not in expanded_gold:
                        expanded_gold.append(tail)
            res = evaluate(pred_ans, expanded_gold)
            return bool(res.get("em", 0.0) >= 1.0)

        return _searchqa_eval
