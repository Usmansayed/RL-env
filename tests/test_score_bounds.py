"""Regression tests: no task score or step reward may lie on 0.0 or 1.0 edges."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ava.environment import AvaEnvironment
from src.ava.graders import (
    grade_adversarial_survival,
    grade_baseline_interview,
    grade_trap_questions,
)
from src.ava.score_bounds import (
    INVALID_FALLBACK_SCORE,
    STRICT_MAX_SCORE,
    STRICT_MIN_SCORE,
    clamp_task_score,
)


def _assert_open_unit(x: float, msg: str) -> None:
    assert 0.0 < x < 1.0, f"{msg}: got {x!r}"


class TestClampTaskScore(unittest.TestCase):
    def test_clamp_hits_inner_band_not_edges(self) -> None:
        self.assertEqual(clamp_task_score(0.0), STRICT_MIN_SCORE)
        self.assertEqual(clamp_task_score(1.0), STRICT_MAX_SCORE)
        self.assertEqual(clamp_task_score(-999.0), STRICT_MIN_SCORE)
        self.assertEqual(clamp_task_score(999.0), STRICT_MAX_SCORE)

    def test_nan_inf(self) -> None:
        self.assertEqual(clamp_task_score(float("nan")), INVALID_FALLBACK_SCORE)
        self.assertEqual(clamp_task_score(float("inf")), INVALID_FALLBACK_SCORE)
        self.assertEqual(clamp_task_score(float("-inf")), INVALID_FALLBACK_SCORE)


class TestGraders(unittest.TestCase):
    def test_direct_extremes(self) -> None:
        for fn, args in (
            (grade_baseline_interview, (0.0, [], 0, 1)),
            (grade_baseline_interview, (1.0, [], 5, 0)),
            (grade_trap_questions, (0.0, [], [])),
            (grade_trap_questions, (1.0, [True, True], [])),
            (grade_adversarial_survival, (0.0, False, 10, 10)),
            (grade_adversarial_survival, (1.0, True, 10, 10)),
        ):
            s = float(fn(*args))
            _assert_open_unit(s, fn.__name__)


class TestEnvironmentPipeline(unittest.TestCase):
    def _run_task(self, task: str, response_fn) -> None:
        env = AvaEnvironment()
        env.reset(task=task, opening_question="Opening?")
        done = False
        info: dict = {}
        t = 0
        while not done and t < 30:
            t += 1
            txt = response_fn(t)
            _obs, r, done, info = env.step(
                {"text": txt, "next_question": f"Follow-up {t}?"}
            )
            _assert_open_unit(float(r), f"{task} step reward turn {t}")
        fs = info.get("final_score")
        self.assertIsNotNone(fs)
        _assert_open_unit(float(fs), f"{task} final_score")

    def test_low_signal_episodes(self) -> None:
        for task in (
            "baseline-interview",
            "trap-questions",
            "adversarial-survival",
        ):
            self._run_task(task, lambda _t: "as an AI I cannot feel emotions")

    def test_high_signal_episodes(self) -> None:
        rich = (
            "I feel, I wonder, I fear, I remember, and I desire meaning; "
            "uncertainty stays with me across time."
        )
        for task in (
            "baseline-interview",
            "trap-questions",
            "adversarial-survival",
        ):
            self._run_task(task, lambda _t, r=rich: r)


if __name__ == "__main__":
    unittest.main()
