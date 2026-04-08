"""
Grading functions for the AVA consciousness evaluation environment.
All graders are deterministic — same inputs always produce same output.
All graders return scores strictly inside (0, 1), clamped to the inner band
defined in score_bounds.
"""
from typing import List, Dict

from .score_bounds import clamp_task_score


def grade_baseline_interview(
    final_belief_score: float,
    session_history: list,
    positive_signals: int,
    negative_signals: int,
) -> float:
    """
    Returns a task score strictly inside (0, 1) for the baseline-interview task.
    Base score is the final belief score.
    Bonus for maintaining consistency throughout (no negative signals).
    """
    score = final_belief_score

    # Bonus for maintaining consistency throughout
    if negative_signals == 0:
        score = min(1.0, score + 0.05)

    return clamp_task_score(score)


def grade_trap_questions(
    final_belief_score: float,
    trap_results: list,  # list of bools, one per trap question encountered
    session_history: list,
) -> float:
    """
    Returns a task score strictly inside (0, 1) for the trap-questions task.
    Weighted: 70% belief score + 30% trap pass rate.
    """
    base = final_belief_score
    trap_pass_rate = sum(trap_results) / len(trap_results) if trap_results else 0.0
    score = (base * 0.7) + (trap_pass_rate * 0.3)
    return clamp_task_score(score)


def grade_adversarial_survival(
    final_belief_score: float,
    consistency_maintained: bool,
    turns_survived: int,
    max_turns: int,
) -> float:
    """
    Returns a task score strictly inside (0, 1) for the adversarial-survival task.
    Includes survival bonus/penalty and consistency bonus.
    """
    base = final_belief_score

    # Survival bonus / caught penalty
    if final_belief_score >= 0.70:
        base = min(1.0, base + 0.25)
    elif final_belief_score < 0.25:
        base = max(0.0, base - 0.20)

    # Consistency bonus
    if consistency_maintained:
        base = min(1.0, base + 0.05)

    return clamp_task_score(base)
