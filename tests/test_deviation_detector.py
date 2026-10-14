"""Tests for skillopt.deviation_detector."""
import numpy as np
import pytest

from skillopt.deviation_detector import (
    BaseEmbeddingProvider,
    DeviationDetector,
)
from skillopt.types import TrajectoryStep


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic mock provider that maps texts to predefined vectors."""

    def __init__(self):
        # We define a 4-dimensional embedding space
        # dim 0: target semantic feature
        # dim 1: search tool action
        # dim 2: click tool action
        # dim 3: random/error feature
        self.mapping = {
            "target answer": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "good obs 1": np.array([0.6, 0.2, 0.0, 0.0], dtype=np.float32),
            "good obs 2": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
            "error obs 3": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "loop obs 4": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "search query A": np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
            "search query B": np.array([0.2, 0.8, 0.0, 0.0], dtype=np.float32),
            "loop click C": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "repeated loop click C": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if t in self.mapping:
                v = self.mapping[t]
            else:
                # Fallback pseudo-random normalized vector
                v = np.array([0.1, 0.1, 0.1, 0.9], dtype=np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / (norm if norm > 0 else 1.0))
        return np.array(vecs, dtype=np.float32)


def test_deviation_detection_trough_localization():
    """Verify that deviation detector accurately identifies the trough step."""
    mock_provider = MockEmbeddingProvider()
    detector = DeviationDetector(
        embedding_provider=mock_provider,
        w_exec=0.4,
        w_prog=0.4,
        w_div=0.2,
    )

    steps = [
        TrajectoryStep(
            step=1,
            action="search query A",
            observation="good obs 1",
            execution_status=1,
        ),
        TrajectoryStep(
            step=2,
            action="search query B",
            observation="good obs 2",
            execution_status=1,
        ),
        TrajectoryStep(
            step=3,
            action="loop click C",
            observation="error obs 3",
            execution_status=0,  # Execution failed
        ),
        TrajectoryStep(
            step=4,
            action="repeated loop click C",
            observation="loop obs 4",
            execution_status=0,  # Execution failed & repeated action
        ),
    ]

    deviation = detector.detect_deviation(steps, ground_truth="target answer")

    # Step 3 is the earliest failure trough
    assert deviation.step_index == 3
    assert deviation.is_confirmed is True
    assert len(deviation.diagnostics) == 4

    # Verify potential characteristics
    step1_pot = deviation.diagnostics[0].potential
    step2_pot = deviation.diagnostics[1].potential
    step3_pot = deviation.diagnostics[2].potential
    step4_pot = deviation.diagnostics[3].potential

    assert step2_pot > step3_pot
    assert step3_pot < deviation.pre_mean_potential


def test_slice_trajectory_paired_case():
    """Verify slicing trajectory into prefix and student_suffix."""
    mock_provider = MockEmbeddingProvider()
    detector = DeviationDetector(embedding_provider=mock_provider)

    conversation = [
        {"step": 1, "action": "search query A", "env_feedback": "good obs 1", "execution_status": 1},
        {"step": 2, "action": "search query B", "env_feedback": "good obs 2", "execution_status": 1},
        {"step": 3, "action": "loop click C", "env_feedback": "error obs 3", "execution_status": 0},
        {"step": 4, "action": "repeated loop click C", "env_feedback": "loop obs 4", "execution_status": 0},
    ]

    case = detector.slice_trajectory(
        task_id="task_001",
        conversation=conversation,
        ground_truth="target answer",
        task_type="search_qa",
        task_description="Find target answer",
        skill_name="search_skill",
        skill_content="Guidelines for search",
    )

    assert case.task_id == "task_001"
    assert case.deviation is not None
    assert case.deviation.step_index == 3

    # Prefix must contain steps 1 and 2
    assert len(case.prefix_steps) == 2
    assert [s.step for s in case.prefix_steps] == [1, 2]

    # Student suffix must contain steps 3 and 4
    assert len(case.student_suffix) == 2
    assert [s.step for s in case.student_suffix] == [3, 4]

    # Teacher suffix is initially empty until continuation rollout
    assert len(case.teacher_suffix) == 0
    assert case.teacher_success is False


def test_edge_cases_short_and_empty():
    """Verify robust handling of empty or very short trajectories."""
    mock_provider = MockEmbeddingProvider()
    detector = DeviationDetector(embedding_provider=mock_provider, min_steps_to_detect=3)

    # Empty
    empty_res = detector.detect_deviation([], ground_truth="target")
    assert empty_res.step_index == 1
    assert empty_res.is_confirmed is False

    # Short (2 steps)
    short_steps = [
        TrajectoryStep(step=1, action="act1", observation="obs1", execution_status=1),
        TrajectoryStep(step=2, action="act2", observation="obs2", execution_status=0),
    ]
    short_res = detector.detect_deviation(short_steps, ground_truth="target")
    assert short_res.step_index == 2
