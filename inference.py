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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Environment variables (NEVER hardcode) ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "")

# LLM judge is mandatory for dialogue generation.
JUDGE_MODEL_NAME = os.environ.get("JUDGE_MODEL_NAME", MODEL_NAME)
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.7"))
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "220"))
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "logs/conversations"))
LOG_FULL_CONVERSATION = os.environ.get("LOG_FULL_CONVERSATION", "1").strip().lower() in (
    "1",
    "true",
    "yes",
)

# --- Task configuration ---
TASKS = ["baseline-interview", "trap-questions", "adversarial-survival"]
BENCHMARK = "ava-consciousness"
MAX_STEPS = {
    "baseline-interview": 8,
    "trap-questions": 10,
    "adversarial-survival": 12,
}
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
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


def run_task(client: OpenAI, env, task_name: str) -> list:
    """
    Run one full episode of the given task.
    Emits [START], [STEP]*, [END] lines to stdout in the required format.
    Returns list of rewards.
    """
    from src.ava.judge import next_question, opening_question

    opening = opening_question(
        client,
        task_name,
        model=JUDGE_MODEL_NAME,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    obs = env.reset(task=task_name, opening_question=opening)

    rewards = []
    done = False
    step_num = 0
    error_val = "null"
    action_text = ""
    transcript_path = _prepare_transcript_file(task_name)

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

    while not done:
        step_num += 1
        question = obs.question

        try:
            messages.append({"role": "user", "content": question})
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
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
                )

            payload = {"text": action_text}
            if next_q is not None:
                payload["next_question"] = next_q

            obs, reward, done, info = env.step(payload)
            error_val = "null"

        except Exception as e:
            action_text = ""
            reward = 0.0
            done = True
            error_val = str(e).replace("\n", " ")[:100]

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

    success = rewards[-1] >= SUCCESS_SCORE_THRESHOLD if rewards else False
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    # [END] line — exactly this format
    print(
        f"[END] success={str(success).lower()} steps={step_num} rewards={rewards_str}",
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
            ),
        )
        print(f"[LOG] conversation_path={transcript_path.as_posix()}", flush=True)

    return rewards


if __name__ == "__main__":
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

    sys.path.insert(0, os.path.dirname(__file__))
    from src.ava.environment import AvaEnvironment

    env = AvaEnvironment()

    for task_name in TASKS:
        run_task(client, env, task_name)
