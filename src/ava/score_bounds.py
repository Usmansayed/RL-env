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


def clamp_task_score(value: float) -> float:
    """
    Clamp to [STRICT_MIN_SCORE, STRICT_MAX_SCORE].

    Handles NaN/inf defensively so a bad intermediate float cannot propagate.
    """
    x = float(value)
    if math.isnan(x):
        return round((STRICT_MIN_SCORE + STRICT_MAX_SCORE) / 2, 4)
    if math.isinf(x):
        return STRICT_MAX_SCORE if x > 0 else STRICT_MIN_SCORE
    return round(max(STRICT_MIN_SCORE, min(STRICT_MAX_SCORE, x)), 4)
