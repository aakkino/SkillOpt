"""Deviation Point Detection module for SkillPivot.

Implements the deviation detection stage from SkillPivot (arXiv:2609.29154):
  1. Computes progress potential for each step t:
     V_t = w1 * Exec_t + w2 * Prog_t + w3 * Div_t
     - Exec_t: Execution validity (deterministic rule on tool outcome, 0 or 1).
     - Prog_t: Positive relevance gain towards task goal / ground truth
               Prog_t = max(0, cos(e(o_t), e(y)) - max_{tau < t} cos(e(o_tau), y))
     - Div_t: Action diversity distance (penalizing repetitive action loops)
              Div_t = 1 - max_{tau < t} cos(e(a_t), e(a_tau))
  2. Identifies deviation point t* at the earliest significant potential trough
     and confirms that subsequent steps remain consistently low.
  3. Supports both external API embeddings (OpenAI/SiliconFlow/DashScope) and
     local SentenceTransformer (all-MiniLM-L6-v2).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from skillopt.types import (
    DeviationPoint,
    PairedSuffixCase,
    StepPotential,
    TrajectoryStep,
)


# ── Embedding Providers (Dual Backend: External API vs. Local) ──────────


class BaseEmbeddingProvider(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a normalized 2D numpy array of embeddings."""


class APIEmbeddingProvider(BaseEmbeddingProvider):
    """External API embedding provider (OpenAI-compatible /v1/embeddings)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = (
            api_key
            or os.environ.get("EMBEDDING_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy-key"
        )
        self.api_base = (
            api_base
            or os.environ.get("EMBEDDING_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)

        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)
        cleaned = [t if t.strip() else " " for t in texts]

        response = client.embeddings.create(input=cleaned, model=self.model)
        vectors = [item.embedding for item in response.data]
        mat = np.array(vectors, dtype=np.float32)

        # L2 normalize
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class LocalMiniLMProvider(BaseEmbeddingProvider):
    """Local SentenceTransformer provider (all-MiniLM-L6-v2)."""

    def __init__(
        self,
        model_name: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        self.model_name = model or model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as err:
                raise ImportError(
                    "The 'sentence-transformers' package is required for the local embedding backend. "
                    "Please install it or configure 'api' embedding backend."
                ) from err

            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 1), dtype=np.float32)

        self._load()
        cleaned = [t if t.strip() else " " for t in texts]
        vecs = self._model.encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)


def get_embedding_provider(
    backend: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseEmbeddingProvider:
    """Factory helper to get the requested or default embedding provider.

    Unified parameter handling:
      - 'backend': 'local' (SentenceTransformer) or 'api'/'openai'/'remote'
      - 'model': maps cleanly to both LocalMiniLMProvider and APIEmbeddingProvider
    """
    chosen = (backend or os.environ.get("EMBEDDING_BACKEND", "local")).lower().strip()
    model_name = model or kwargs.pop("model_name", None)

    if chosen in {"api", "openai", "remote"}:
        return APIEmbeddingProvider(model=model_name, **kwargs)

    if chosen in {"local", "minilm", "sentence_transformers"}:
        return LocalMiniLMProvider(model=model_name, **kwargs)

    if chosen in {"auto"}:
        try:
            return LocalMiniLMProvider(model=model_name, **kwargs)
        except ImportError:
            return APIEmbeddingProvider(model=model_name, **kwargs)

    raise ValueError(
        f"Unsupported embedding backend '{chosen}'. Expected 'local', 'api', or 'auto'."
    )


# ── Deviation Point Detector Core ────────────────────────────────────────


class DeviationDetector:
    """Deviation Point Detector based on Progress Potential and Trough Analysis.

    Computes:
      V_t = w_exec * Exec_t + w_prog * Prog_t + w_div * Div_t
    and determines deviation point t*.
    """

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider | None = None,
        w_exec: float = 0.4,
        w_prog: float = 0.4,
        w_div: float = 0.2,
        trough_threshold_ratio: float = 0.75,
        min_steps_to_detect: int = 3,
    ):
        self.provider = embedding_provider or get_embedding_provider()
        self.w_exec = w_exec
        self.w_prog = w_prog
        self.w_div = w_div
        self.trough_threshold_ratio = trough_threshold_ratio
        self.min_steps_to_detect = min_steps_to_detect

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two 1D normalized vectors."""
        val = float(np.dot(a, b))
        return max(-1.0, min(1.0, val))

    def detect_deviation(
        self,
        steps: list[TrajectoryStep],
        ground_truth: str = "",
    ) -> DeviationPoint:
        """Analyze trajectory steps and return the detected DeviationPoint."""
        n = len(steps)
        if n == 0:
            return DeviationPoint(
                step_index=1,
                trough_potential=0.0,
                pre_mean_potential=0.0,
                post_mean_potential=0.0,
                is_confirmed=False,
                reason="Empty trajectory",
                diagnostics=[],
            )

        if n < self.min_steps_to_detect:
            # Trajectory too short: deviation set to the last step by default
            potentials = [
                StepPotential(
                    step=s.step,
                    exec_score=float(s.execution_status),
                    prog_score=0.0,
                    div_score=1.0,
                    potential=float(s.execution_status),
                    is_trough=False,
                )
                for s in steps
            ]
            return DeviationPoint(
                step_index=steps[-1].step,
                trough_potential=potentials[-1].potential,
                pre_mean_potential=potentials[0].potential,
                post_mean_potential=potentials[-1].potential,
                is_confirmed=False,
                reason=f"Trajectory too short ({n} < {self.min_steps_to_detect})",
                diagnostics=potentials,
            )

        # 1. Prepare texts for vector encoding
        actions = [s.action for s in steps]
        observations = [s.observation for s in steps]
        target_text = ground_truth.strip() if ground_truth.strip() else "target objective"

        # Batch embed: [observations (n)] + [ground_truth (1)] + [actions (n)]
        all_texts = observations + [target_text] + actions
        embeddings = self.provider.embed(all_texts)

        obs_vecs = embeddings[:n]
        target_vec = embeddings[n]
        act_vecs = embeddings[n + 1 :]

        # 2. Step-by-step potential calculation
        diagnostics: list[StepPotential] = []
        hist_sim_max = 0.0

        for i, step in enumerate(steps):
            # (a) Exec_t
            exec_t = float(step.execution_status)

            # (b) Prog_t
            sim_y = self._cosine_similarity(obs_vecs[i], target_vec)
            if i == 0:
                prog_t = max(0.0, sim_y)
                hist_sim_max = sim_y
            else:
                prog_t = max(0.0, sim_y - hist_sim_max)
                if sim_y > hist_sim_max:
                    hist_sim_max = sim_y

            # (c) Div_t: 1 - max_{tau < t} cos(a_t, a_tau)
            if i == 0:
                div_t = 1.0
            else:
                max_sim_prev = max(
                    self._cosine_similarity(act_vecs[i], act_vecs[j])
                    for j in range(i)
                )
                div_t = max(0.0, 1.0 - max_sim_prev)

            # Total potential V_t
            v_t = self.w_exec * exec_t + self.w_prog * prog_t + self.w_div * div_t

            diagnostics.append(
                StepPotential(
                    step=step.step,
                    exec_score=round(exec_t, 4),
                    prog_score=round(prog_t, 4),
                    div_score=round(div_t, 4),
                    potential=round(v_t, 4),
                )
            )

        # 3. Deviation Point Trough Localization (arXiv:2609.29154 Sec. 3.4)
        scores = [d.potential for d in diagnostics]

        # Candidate is the lowest score step (skip step 1 if n >= 4 to preserve basic prefix)
        start_idx = 1 if n >= 4 else 0
        min_cand_idx = start_idx + int(np.argmin(scores[start_idx:]))

        # Identify the earliest step that initiated this sustained failure trough:
        # If an earlier step in [start_idx, min_cand_idx] already exhibited invalid execution (exec==0)
        # and subsequent steps failed to recover, anchor deviation at that earliest turning point.
        selected_idx = min_cand_idx
        for idx in range(start_idx, min_cand_idx):
            if diagnostics[idx].exec_score == 0.0:
                selected_idx = idx
                break

        min_idx = selected_idx
        pre_scores = scores[:min_idx] if min_idx > 0 else [scores[0]]
        post_scores = scores[min_idx:]

        pre_mean = float(np.mean(pre_scores))
        post_mean = float(np.mean(post_scores))

        diagnostics[min_idx].is_trough = True

        # Confirmation condition: trough potential is significantly below pre-mean or invalid execution
        is_confirmed = (
            (scores[min_idx] <= pre_mean * self.trough_threshold_ratio or diagnostics[min_idx].exec_score == 0.0)
            and post_mean <= pre_mean
        )

        reason = (
            f"Deviation trough detected at step {steps[min_idx].step} (V={scores[min_idx]:.3f}); "
            f"Pre-mean V={pre_mean:.3f}, Post-mean V={post_mean:.3f}"
        )

        return DeviationPoint(
            step_index=steps[min_idx].step,
            trough_potential=scores[min_idx],
            pre_mean_potential=round(pre_mean, 4),
            post_mean_potential=round(post_mean, 4),
            is_confirmed=is_confirmed,
            reason=reason,
            diagnostics=diagnostics,
        )

    def slice_trajectory(
        self,
        task_id: str,
        conversation: list[dict],
        ground_truth: str = "",
        *,
        task_type: str = "",
        task_description: str = "",
        skill_name: str = "",
        skill_content: str = "",
    ) -> PairedSuffixCase:
        # Filter out system and evaluation records (e.g., SearchQA eval summary)
        valid_records = [
            raw for raw in conversation
            if isinstance(raw, dict) and not TrajectoryStep.is_system_or_eval(raw)
        ]
        if not valid_records:
            valid_records = conversation

        steps = [
            TrajectoryStep.from_dict(raw, default_step=idx + 1)
            for idx, raw in enumerate(valid_records)
        ]
        deviation = self.detect_deviation(steps, ground_truth=ground_truth)

        return PairedSuffixCase.slice_from_full_trajectory(
            task_id=task_id,
            full_steps=steps,
            deviation=deviation,
            task_type=task_type,
            task_description=task_description,
            ground_truth=ground_truth,
            skill_name=skill_name,
            skill_content=skill_content,
        )
