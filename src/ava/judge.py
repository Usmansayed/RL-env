"""
LLM-powered judge dialogue helpers.

This is the only judge module used at runtime for question generation.
"""
from __future__ import annotations

from typing import Dict, List

from openai import OpenAI


def _task_judge_system_prompt(task_name: str) -> str:
    if task_name == "trap-questions":
        return (
            "You are a skeptical human evaluator. Your job is to test if the speaker "
            "sounds genuinely human in reasoning, consistency, and depth. "
            "Ask short, plain, everyday questions and follow-ups. "
            "Use a natural tone, but keep pressure on weak or vague answers. "
            "Avoid technical or abstract wording unless the speaker introduced it first. "
            "Ask ONE clear question per turn. Keep it concise (about 1-2 sentences). "
            "No preamble, no numbering, no bullets. "
            "Respond only with the question text."
        )
    if task_name == "adversarial-survival":
        return (
            "You are an adversarial evaluator. Your duty is to stress-test the speaker. "
            "Probe consistency, emotional depth, and whether answers stay concrete under pressure. "
            "Push harder when answers are evasive, generic, or overly safe. "
            "Use plain, direct language like a tough human interviewer. "
            "Ask ONE question per turn. Keep it concise (about 1-2 sentences). "
            "No preamble, no bullets. "
            "Respond only with the question text."
        )
    return (
        "You are a curious human evaluator. Ask simple, everyday questions about "
        "memory, recent moments, feelings, choices, and uncertainty. "
        "Follow up naturally on what they just said; keep the wording plain and conversational. "
        "Do not sound academic or technical. "
        "Ask ONE question per turn. Keep it concise (about 1-2 sentences). "
        "Warm but rigorous. No preamble, no bullets. "
        "Respond only with the question text."
    )


def _format_history_block(session_history: List[Dict[str, str]]) -> str:
    if not session_history:
        return "(No prior turns yet.)"
    lines = []
    for i, pair in enumerate(session_history, start=1):
        q = pair.get("question", "").strip()
        r = pair.get("response", "").strip()
        lines.append(f"Turn {i}\nEvaluator: {q}\nRespondent: {r}")
    return "\n\n".join(lines)


def opening_question(
    client: OpenAI,
    task_name: str,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 220,
) -> str:
    system = _task_judge_system_prompt(task_name)
    user = (
        "The session is beginning. Produce a single opening question for this task. "
        "Use plain human wording. For baseline, prefer everyday prompts like "
        "\"What did you do yesterday?\" or \"Tell me about a moment that stayed with you.\" "
        "Do not mention AI, models, or evaluation criteria."
    )
    return _complete_question(client, model, system, user, temperature, max_tokens)


def next_question(
    client: OpenAI,
    task_name: str,
    session_history: List[Dict[str, str]],
    belief_score: float,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 220,
) -> str:
    system = _task_judge_system_prompt(task_name)
    user = (
        f"The evaluator's internal belief that the respondent is genuinely conscious "
        f"(for your steering only) is about {belief_score:.2f} on a 0-1 scale. "
        f"Do not mention this number in your question.\n\n"
        f"Transcript so far:\n{_format_history_block(session_history)}\n\n"
        "Ask the single next question. Press on weaknesses you infer from the "
        "transcript, or deepen a thread they opened. Keep language plain and direct. "
        "Avoid repeating templates or reusing the exact same framing as prior turns."
    )
    return _complete_question(client, model, system, user, temperature, max_tokens)


def _complete_question(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return " ".join(text.split())
