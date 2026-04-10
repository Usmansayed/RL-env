"""
Validator-safe bounds for task scores and externally visible rewards.

Phase-2 style checks require each task score to lie strictly in (0, 1) — not
exactly 0.0 or 1.0. AVA uses a fixed inner band everywhere scores or rewards
are exposed to clients, stdout, or metadata.
"""
from __future__ import annotations

import math

# Inner band inside (0, 1); keeps float noise from snapping to exact edges.
STRICT_MIN_SCORE = 0.1
STRICT_MAX_SCORE = 0.99
OUTPUT_EPSILON = 1e-6
INVALID_FALLBACK_SCORE = 0.5


def _safe_float(value: object, default: float = INVALID_FALLBACK_SCORE) -> float:
    """Best-effort float conversion with fallback for invalid inputs."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(x) or math.isinf(x):
        return float(default)
    return x


def clamp_task_score(value: object) -> float:
    """
    Clamp to [STRICT_MIN_SCORE, STRICT_MAX_SCORE].

    Handles NaN/inf defensively so a bad intermediate float cannot propagate.
    """
    x = _safe_float(value)
    return round(max(STRICT_MIN_SCORE, min(STRICT_MAX_SCORE, x)), 4)


def harden_score_for_output(value: object) -> float:
    """
    Hard safety layer used for final payload/output surfaces only.
    Produces a strict-open score in (1e-6, 1 - 1e-6), defaulting invalid
    values to 0.5 before clamping.
    """
    x = _safe_float(value)
    return max(OUTPUT_EPSILON, min(1.0 - OUTPUT_EPSILON, x))


def is_strict_open_score(value: object) -> bool:
    """Return True only when value is a valid finite float in (0, 1)."""
    x = _safe_float(value, default=float("nan"))
    return not math.isnan(x) and (0.0 < x < 1.0)
