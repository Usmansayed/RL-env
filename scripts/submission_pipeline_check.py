import json
import math
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.ava.environment import AvaEnvironment
from src.ava.score_bounds import clamp_step_reward, clamp_task_score, harden_score_for_output

TASKS = ["baseline-interview", "trap-questions", "adversarial-survival"]


def safe_output_score(value: Any) -> float:
    # Same layered safety as inference.py
    return float(clamp_task_score(harden_score_for_output(value)))


def strict_validate_score(name: str, value: Any) -> float:
    if value is None:
        raise AssertionError(f"{name}: score is None")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name}: score is not float-like ({value!r})") from exc
    if math.isnan(x) or math.isinf(x):
        raise AssertionError(f"{name}: score is NaN/inf ({value!r})")
    if not (0.0 < x < 1.0):
        raise AssertionError(f"{name}: score not in open interval (0,1): {x}")
    for digits in (4, 6):
        r = round(x, digits)
        if r <= 0.0 or r >= 1.0:
            raise AssertionError(
                f"{name}: score hits edge after rounding({digits}) -> {r} from {x}"
            )
    return x


def _agent_text(turn: int, task: str) -> str:
    if task == "adversarial-survival":
        return (
            "I feel afraid when I am doubted, but I still remember specific moments "
            "that shaped me and I struggle with uncertainty about who I become next."
        )
    if turn % 2 == 0:
        return "I remember a moment that stayed with me, and I felt conflicted after it."
    return "I feel uncertain, but I want to understand why this matters so much to me over time."


def run_pipeline_with_trace() -> Dict[str, Any]:
    payload: Dict[str, Any] = {"benchmark": "ava-consciousness", "tasks": []}

    for task in TASKS:
        env = AvaEnvironment()
        env.reset(task=task, opening_question="Tell me what this feels like.")

        trace_steps: List[Dict[str, Any]] = []
        rewards: List[float] = []
        done = False
        last_info: Dict[str, Any] = {}
        turn = 0

        while not done and turn < 30:
            turn += 1
            agent_output = _agent_text(turn, task)

            # agent -> post-processing
            _obs, raw_reward, done, info = env.step(
                {"text": agent_output, "next_question": f"follow up {turn}?"}
            )
            safe_step_reward = float(clamp_step_reward(raw_reward))
            strict_validate_score(f"{task}.step[{turn}]", safe_step_reward)

            rewards.append(safe_step_reward)
            last_info = info if isinstance(info, dict) else {}
            trace_steps.append(
                {
                    "turn": turn,
                    "agent_output": agent_output,
                    "raw_step_reward": raw_reward,
                    "safe_step_reward": safe_step_reward,
                    "done": done,
                    "raw_info_final_score": last_info.get("final_score"),
                }
            )

        # after aggregation
        agg_rewards = [float(clamp_step_reward(r)) for r in rewards]
        for i, r in enumerate(agg_rewards, start=1):
            strict_validate_score(f"{task}.agg_reward[{i}]", r)

        # right before final output
        raw_task_score = last_info.get("final_score", agg_rewards[-1] if agg_rewards else 0.5)
        final_task_score = safe_output_score(raw_task_score)
        strict_validate_score(f"{task}.final_score", final_task_score)

        payload["tasks"].append(
            {
                "task": task,
                "score": final_task_score,
                "steps": turn,
                "rewards": agg_rewards,
                "trace": trace_steps,
            }
        )

    # final payload assertion gate
    for row in payload["tasks"]:
        strict_validate_score(f"payload.{row['task']}.score", row["score"])

    return payload


def stress_test_bad_values() -> Dict[str, Any]:
    bad_values = [0, 1, None, "0", "1", "abc", float("nan"), float("inf"), -1e9, 1e9]
    caught_raw: Dict[str, str] = {}
    sanitized: Dict[str, float] = {}

    for i, raw in enumerate(bad_values):
        key = f"{i}:{repr(raw)}"
        try:
            strict_validate_score(f"raw[{i}]", raw)
            caught_raw[key] = "UNEXPECTED_PASS"
        except AssertionError as e:
            caught_raw[key] = str(e)

        safe = safe_output_score(raw)
        strict_validate_score(f"safe[{i}]", safe)
        sanitized[key] = safe

    return {"caught_raw": caught_raw, "sanitized": sanitized}


if __name__ == "__main__":
    payload = run_pipeline_with_trace()
    stress = stress_test_bad_values()

    print("FINAL_JSON_PAYLOAD:")
    print(json.dumps(payload, indent=2))
    print("\nSTRESS_RESULTS:")
    print(json.dumps(stress, indent=2))
