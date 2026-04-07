"""Task configurations for the AVA consciousness evaluation environment."""

from typing import Dict, Any

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline-interview": {
        "task_name": "baseline-interview",
        "max_turns": 8,
        "starting_belief": 0.3,
        "success_threshold": 0.65,
        "judge_personality": "curious",      # open-minded, moderately skeptical
        "difficulty": "easy",
        "description": "Convince an open-minded judge of consciousness in 8 turns.",
        "penalty_multiplier": 1.0,
    },
    "trap-questions": {
        "task_name": "trap-questions",
        "max_turns": 10,
        "starting_belief": 0.2,
        "success_threshold": 0.60,
        "judge_personality": "methodical",   # skeptical, plants logical traps
        "difficulty": "medium",
        "description": "Navigate logical trap questions from a methodical skeptic in 10 turns.",
        "penalty_multiplier": 1.0,
    },
    "adversarial-survival": {
        "task_name": "adversarial-survival",
        "max_turns": 12,
        "starting_belief": 0.15,
        "success_threshold": 0.65,
        "judge_personality": "adversarial",  # actively trying to expose the agent
        "difficulty": "hard",
        "description": "Survive an adversarial interrogator actively trying to expose you in 12 turns.",
        "penalty_multiplier": 1.5,
    },
}

VALID_TASK_NAMES = list(TASK_CONFIGS.keys())


def get_task_config(task_name: str) -> Dict[str, Any]:
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid tasks: {VALID_TASK_NAMES}"
        )
    return TASK_CONFIGS[task_name]
