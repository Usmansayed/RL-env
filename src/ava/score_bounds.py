"""
Validator-safe bounds for task scores and externally visible rewards.

Phase-2 checks often require each task grade strictly in (0, 1), excluding
exact 0.0 / 1.0 and sometimes rejecting boundary float noise. AVA uses:
  - Task / belief / final_score: [0.001, 0.999] (clamp_task_score)
  - Per-step rewards (stdout uses .2f): [0.01, 0.99] (clamp_step_reward)
"""
from __future__ import annotations

import math

# Final task grades, belief, info["final_score"] — Discord / strict-open band.
STRICT_MIN_SCORE = 0.001
STRICT_MAX_SCORE = 0.999

# Step rewards logged with two decimal places must stay off 0.00 / 1.00.
STEP_REWARD_MIN = 0.01
STEP_REWARD_MAX = 0.99

OUTPUT_EPSILON = 1e-9
INVALID_FALLBACK_SCORE = 0.5


def _safe_float(value: object, default: float = INVALID_FALLBACK_SCORE) -> float:
    """Best-effort float conversion with fallback for invalid inputs."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(x) or math.isinf(x):
        return float(default)
    return float(x)


def clamp_task_score(value: object) -> float:
    """
    Clamp task-grade surfaces to [STRICT_MIN_SCORE, STRICT_MAX_SCORE].

    Returns a plain Python float. Handles NaN/inf defensively.
    """
    x = _safe_float(value)
    return float(round(max(STRICT_MIN_SCORE, min(STRICT_MAX_SCORE, x)), 6))


def clamp_step_reward(value: object) -> float:
    """Clamp per-step reward for APIs and [STEP] lines (two-decimal safe band)."""
    x = _safe_float(value)
    return float(round(max(STEP_REWARD_MIN, min(STEP_REWARD_MAX, x)), 4))


def harden_score_for_output(value: object) -> float:
    """
    Hard safety layer before clamp_task_score: finite float squeezed off 0/1 edges.
    """
    x = _safe_float(value)
    return float(max(OUTPUT_EPSILON, min(1.0 - OUTPUT_EPSILON, x)))


def is_strict_open_score(value: object) -> bool:
    """Return True only when value is a valid finite float in (0, 1)."""
    x = _safe_float(value, default=float("nan"))
    return not math.isnan(x) and (0.0 < x < 1.0)
