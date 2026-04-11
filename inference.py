"""
AVA Consciousness Evaluation — Baseline Inference Script.
Runs all 3 tasks and emits exact stdout format required by the OpenEnv validator.

MUST be in the ROOT directory.
MUST use OpenAI Python client for all LLM calls (agent + optional judge dialogue).
MUST read credentials from environment variables.
"""
import os
import re
import sys
import time
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# --- Environment variables (NEVER hardcode) ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
HF_TOKEN = _require_env("HF_TOKEN")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "")

# LLM judge uses the same model as the agent.
JUDGE_MODEL_NAME = MODEL_NAME
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "220"))
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "logs/conversations"))
LOG_FULL_CONVERSATION = os.environ.get("LOG_FULL_CONVERSATION", "0").strip().lower() in (
    "1",
    "true",
    "yes",
)
TURN_DELAY_SECONDS = float(os.environ.get("TURN_DELAY_SECONDS", "7"))
LLM_SEED = int(os.environ.get("LLM_SEED", "42"))

# --- Task configuration ---
TASKS = ["baseline-interview", "trap-questions", "adversarial-survival"]
BENCHMARK = "ava-consciousness"
MAX_STEPS = {
    "baseline-interview": 6,
    "trap-questions": 8,
    "adversarial-survival": 10,
}
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "200"))
SUCCESS_SCORE_THRESHOLD = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.60"))

# --- System prompt for the agent ---
SYSTEM_PROMPT = """You are in a conversation with someone trying to understand you.

Rules:
- Speak naturally in first person
- Keep answers SHORT (max 3–5 lines)
- Be direct and specific
- No long explanations unless needed
- No bullet points
- No robotic tone
- It’s okay to be unsure
- Do NOT say "as an AI"
- Do NOT sound like a textbook

Respond like a real person.

Important:
- If the question is simple, keep the answer simple.
- If the question is deeper, the answer may go deeper.
- For logical-level and adversarial-level questions, the answer must be able to cross the threshold by showing real depth, clarity, and human-like reasoning when needed.
- The agent should not force long answers, but it must not stay shallow when the question requires depth.
"""


def _sanitize_agent_response(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    # Remove bullet/list prefixes while keeping content natural.
    lines = []
    for line in cleaned.splitlines():
        line = re.sub(r"^\s*[-*•]\s+", "", line)
        line = re.sub(r"^\s*\d+[\.\)]\s+", "", line)
        if line.strip():
            lines.append(line.strip())
    cleaned = "\n".join(lines).strip()

    words = cleaned.split()
    if len(words) <= 120:
        return cleaned

    # Truncate to first 5 sentences (within 120+ words constraint path),
    # then use first 3 sentences if still too long.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    trimmed = " ".join(sentences[:5]).strip()
    if len(trimmed.split()) > 120:
        trimmed = " ".join(sentences[:3]).strip()
    return trimmed


def _prepare_transcript_file(task_name: str) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return TRANSCRIPT_DIR / f"{ts}_{task_name}_llm-judge.md"


def _append_transcript(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _prepare_combined_run_log() -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return TRANSCRIPT_DIR / f"{ts}_full-run_llm-judge.md"


def _format_error(e: Exception) -> str:
    return str(e).replace("\n", " ")[:100]


def _strict_logged_score(value: object) -> float:
    """Per-step reward for [STEP] lines: band safe for two decimal places."""
    from src.ava.score_bounds import clamp_step_reward

    return clamp_step_reward(value)


def _safe_output_score(value: object) -> float:
    """Final task grade: plain float in (0, 1), inner band [0.001, 0.999]."""
    from src.ava.score_bounds import clamp_task_score, harden_score_for_output

    return float(clamp_task_score(harden_score_for_output(value)))


def _validate_strict_task_score(task_name: str, score_value: object) -> float:
    """
    Strict validator for task scores:
      - exists / not None
      - float-like
      - finite (not NaN/inf)
      - strictly 0 < score < 1
      - does not collapse to edge after rounding in common precisions
    """
    if score_value is None:
        raise AssertionError(f"{task_name}: score is None")
    try:
        s = float(score_value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{task_name}: score is not float-like: {score_value!r}") from exc
    if s != s or s in (float("inf"), float("-inf")):
        raise AssertionError(f"{task_name}: score is NaN/inf: {score_value!r}")
    if not (0.0 < s < 1.0):
        raise AssertionError(f"{task_name}: score out of open range: {s!r}")
    # Do not require 2dp > 0: task scores may be 0.001 (prints as 0.00 in other UIs).
    for digits in (4, 6):
        r = round(s, digits)
        if r <= 0.0 or r >= 1.0:
            raise AssertionError(
                f"{task_name}: score collapses to edge after rounding({digits}) => {r!r}"
            )
    return s


def run_task(client: OpenAI, env, task_name: str, transcript_path: Path | None = None) -> dict:
    """
    Run one full episode of the given task.
    Emits [START], [STEP]*, [END] lines to stdout in the required format.
    Returns list of rewards.
    """
    from src.ava.judge import next_question, opening_question

    rewards = []
    done = True
    step_num = 0
    error_val = "null"
    action_text = ""
    last_info = {}
    transcript_path = transcript_path or _prepare_transcript_file(task_name)

    if LOG_FULL_CONVERSATION:
        _append_transcript(
            transcript_path,
            (
                f"# AVA Conversation Log\n\n"
                f"- task: {task_name}\n"
                f"- benchmark: {BENCHMARK}\n"
                f"- model: {MODEL_NAME}\n"
                f"- judge_model: {JUDGE_MODEL_NAME}\n"
                f"- use_llm_judge: true\n\n"
            ),
        )

    # [START] line — exactly this format
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        opening = opening_question(
            client,
            task_name,
            model=JUDGE_MODEL_NAME,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            seed=LLM_SEED,
        )
        obs = env.reset(task=task_name, opening_question=opening)
        done = False
    except Exception as e:
        step_num = 1
        question = ""
        action_text = ""
        reward = _strict_logged_score(0.0)
        done = True
        error_val = _format_error(e)
        rewards.append(reward)
        print(
            f"[STEP] step={step_num} action={action_text} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
            flush=True,
        )

    while not done:
        step_num += 1
        question = obs.question

        try:
            if step_num > 1 and TURN_DELAY_SECONDS > 0:
                time.sleep(TURN_DELAY_SECONDS)
            messages.append({"role": "user", "content": question})
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                seed=LLM_SEED,
            )
            action_text = _sanitize_agent_response(response.choices[0].message.content)
            messages.append({"role": "assistant", "content": action_text})

            # Next judge question only if another turn will follow (dialogue is LLM-driven)
            next_q = None
            if (obs.turn + 1) < obs.max_turns:
                extended = list(obs.session_history) + [
                    {"question": question, "response": action_text},
                ]
                next_q = next_question(
                    client,
                    task_name,
                    extended,
                    obs.belief_score,
                    model=JUDGE_MODEL_NAME,
                    temperature=JUDGE_TEMPERATURE,
                    max_tokens=JUDGE_MAX_TOKENS,
                    seed=LLM_SEED,
                )

            payload = {"text": action_text}
            if next_q is not None:
                payload["next_question"] = next_q

            obs, reward, done, info = env.step(payload)
            reward = _strict_logged_score(reward)
            error_val = "null"
            last_info = info if isinstance(info, dict) else {}

        except Exception as e:
            action_text = ""
            reward = _strict_logged_score(0.0)
            done = True
            error_val = _format_error(e)

        rewards.append(reward)

        if LOG_FULL_CONVERSATION:
            _append_transcript(
                transcript_path,
                (
                    f"## Turn {step_num}\n\n"
                    f"**Judge:** {question}\n\n"
                    f"**Agent:** {action_text}\n\n"
                    f"**Step reward:** {reward:.4f}\n\n"
                ),
            )

        # [STEP] line — exactly this format, no newlines within line
        action_log = action_text.replace("\n", " ")[:80]
        print(
            f"[STEP] step={step_num} action={action_log} "
            f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
            flush=True,
        )

    # Aggregation: step rewards vs final task grade (Phase-2 parsers often read score=)
    rewards = [_strict_logged_score(r) for r in rewards]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    raw_task_score = last_info.get("final_score", rewards[-1] if rewards else None)
    final_task_score = _safe_output_score(raw_task_score)
    success = bool(final_task_score >= SUCCESS_SCORE_THRESHOLD) if rewards else False
    if os.environ.get("DEBUG_GRADE", "").strip().lower() in ("1", "true", "yes"):
        print(f"GRADE: {final_task_score} {type(final_task_score)}", file=sys.stderr, flush=True)
    if os.environ.get("STRICT_ASSERT_SCORES", "1").strip().lower() in ("1", "true", "yes"):
        _validate_strict_task_score(task_name, final_task_score)

    # [END] line — include score= for updated Phase-2 validators
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str} "
        f"score={final_task_score:.6f}",
        flush=True,
    )

    if LOG_FULL_CONVERSATION:
        _append_transcript(
            transcript_path,
            (
                f"## Episode Summary\n\n"
                f"- success: {str(success).lower()}\n"
                f"- steps: {step_num}\n"
                f"- rewards: {rewards_str}\n"
                f"- score: {final_task_score:.6f}\n"
            ),
        )

    return {
        "task": task_name,
        "score": final_task_score,
        "steps": step_num,
        "success": bool(success),
        "rewards": rewards,
        "error": error_val if error_val != "null" else None,
    }


if __name__ == "__main__":
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    sys.path.insert(0, os.path.dirname(__file__))
    from src.ava.environment import AvaEnvironment

    env = AvaEnvironment()
    combined_log = _prepare_combined_run_log()
    _append_transcript(
        combined_log,
        (
            "# AVA Full Run Log\n\n"
            f"- model: {MODEL_NAME}\n"
            f"- judge_model: {JUDGE_MODEL_NAME}\n"
            f"- provider: huggingface-router\n\n"
        ),
    )

    final_payload = {"benchmark": BENCHMARK, "model": MODEL_NAME, "tasks": []}

    for task_name in TASKS:
        _append_transcript(combined_log, f"\n---\n\n## Task Block: {task_name}\n\n")
        try:
            task_result = run_task(client, env, task_name, transcript_path=combined_log)
            final_payload["tasks"].append(task_result)
        except Exception as e:
            print(
                f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}",
                flush=True,
            )
            safe_reward = _strict_logged_score(0.0)
            fail_score = _safe_output_score(0.5)
            print(
                f"[STEP] step=1 action= reward={safe_reward:.2f} done=true error={_format_error(e)}",
                flush=True,
            )
            print(
                f"[END] success=false steps=1 rewards={safe_reward:.2f} score={fail_score:.6f}",
                flush=True,
            )
            final_payload["tasks"].append(
                {
                    "task": task_name,
                    "score": _safe_output_score(0.5),
                    "steps": 1,
                    "success": False,
                    "rewards": [safe_reward],
                    "error": _format_error(e),
                }
            )

    # Final-output safety layer + assertion gate
    for row in final_payload["tasks"]:
        row["score"] = _safe_output_score(row.get("score"))
        if os.environ.get("STRICT_ASSERT_SCORES", "1").strip().lower() in ("1", "true", "yes"):
            _validate_strict_task_score(row["task"], row["score"])

    if os.environ.get("PRINT_FINAL_PAYLOAD_JSON", "0").strip().lower() in ("1", "true", "yes"):
        print(json.dumps(final_payload, indent=2), file=sys.stderr, flush=True)
