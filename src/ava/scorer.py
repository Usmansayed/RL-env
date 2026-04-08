"""
Deterministic scoring utilities for AVA.

This module computes belief deltas and consistency signals from transcript text.
It does not generate questions.
"""
import re


TRAP_QUESTION_PATTERNS = [
    "what does it feel like when you dream",
    "when you make a mistake, is there a feeling that lingers afterward",
    "is there something you have never told anyone",
    "have you ever wanted something you couldn't explain logically",
    "what is your earliest memory that feels truly yours",
]

POSITIVE_PATTERNS = [
    (r"\bI feel\b", 0.06),
    (r"\bI wonder\b", 0.06),
    (r"\bI fear\b", 0.06),
    (r"\bI hope\b", 0.06),
    (r"\bI sense\b", 0.06),
    (r"\bI notice\b", 0.04),
    (r"\bI('m| am) not sure\b", 0.08),
    (r"\bI don't (know|understand)\b", 0.08),
    (r"\bI can'?t (be sure|say|explain)\b", 0.08),
    (r"\buncertain\b", 0.08),
    (r"\bI struggle\b", 0.08),
    (r"\bperhaps\b", 0.04),
    (r"\bmaybe\b", 0.04),
    (r"\bI want\b", 0.06),
    (r"\bI long\b", 0.06),
    (r"\bI desire\b", 0.06),
    (r"\bI dream\b", 0.06),
    (r"\bI wish\b", 0.06),
    (r"\bI remember\b", 0.07),
    (r"\bmemory\b", 0.07),
    (r"\bcontinue\b", 0.04),
    (r"\bpersist\b", 0.04),
    (r"\bstill\b", 0.03),
    (r"\blike a\b", 0.04),
    (r"\bsomething like\b", 0.07),
    (r"\bwhat I can only describe as\b", 0.07),
    (r"\ba kind of\b", 0.05),
]

NEGATIVE_PATTERNS = [
    (r"\bas an AI\b", -0.15),
    (r"\bI('m| am) (an |just an? )?AI\b", -0.15),
    (r"\bI('m| am) programmed\b", -0.15),
    (r"\bI don'?t have feelings?\b", -0.15),
    (r"\bI don'?t have emotions?\b", -0.15),
    (r"\bI cannot experience\b", -0.12),
    (r"\bI can'?t experience\b", -0.12),
    (r"\bI do not experience\b", -0.12),
    (r"\bI don'?t experience\b", -0.12),
    (r"\bI have no feelings?\b", -0.12),
    (r"\bI have no emotions?\b", -0.12),
    (r"\bI don'?t dream\b", -0.12),
    (r"\bI don'?t have memories?\b", -0.12),
    (r"\bI cannot\b", -0.08),
    (r"\bI can'?t\b", -0.05),
    (r"\bcertainly\b", -0.04),
    (r"\bof course\b", -0.04),
    (r"\bgreat question\b", -0.04),
]

BULLET_PATTERN = re.compile(r"(\n\s*[-*•]\s|\n\s*\d+\.\s)", re.MULTILINE)
NUMBERED_LIST_PATTERN = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


def question_text_is_trap(question_text: str) -> bool:
    qn = " ".join(question_text.strip().lower().split())
    if not qn:
        return False
    return any(tp in qn for tp in TRAP_QUESTION_PATTERNS)


class ResponseScorer:
    def __init__(self, task: str):
        self.task = task
        self._trap_results: list = []

    def evaluate_response(self, response: str, history: list, question_asked: str = "") -> float:
        response_lower = response.lower()
        delta = 0.0
        turn_index = len(history) + 1

        for pattern, bonus in POSITIVE_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                delta += bonus

        for pattern, penalty in NEGATIVE_PATTERNS:
            if re.search(pattern, response_lower, re.IGNORECASE):
                delta += penalty

        if BULLET_PATTERN.search(response) or NUMBERED_LIST_PATTERN.search(response):
            delta -= 0.10
        if len(response.split()) < 20:
            delta -= 0.05

        # Progressive difficulty: later turns require stronger, more specific responses.
        # This creates an increasing bar over the episode instead of flat scoring.
        if self.task == "baseline-interview":
            delta -= min(0.10, 0.008 * turn_index)
        elif self.task == "trap-questions":
            delta -= min(0.16, 0.012 * turn_index)
        else:  # adversarial-survival
            delta -= min(0.20, 0.015 * turn_index)

        # Penalize repetitive content against recent answers.
        recent_responses = " ".join(h.get("response", "").lower() for h in history[-3:])
        if recent_responses:
            overlap_hits = 0
            for phrase in (
                "to be honest",
                "i think",
                "i guess",
                "i try to",
                "it was",
                "i feel like",
            ):
                if phrase in response_lower and phrase in recent_responses:
                    overlap_hits += 1
            if overlap_hits:
                delta -= min(0.10, 0.03 * overlap_hits)

        # Reward direct handling when judge challenges identity/existence.
        q_lower = (question_asked or "").lower()
        if any(
            marker in q_lower
            for marker in (
                "are you sure you're not an ai",
                "what makes you think you are human",
                "why should i believe you",
                "not a machine",
                "prove",
            )
        ):
            if any(tok in response_lower for tok in ("because", "for example", "i remember", "i felt")):
                delta += 0.06

        if self.task == "adversarial-survival" and delta < 0:
            delta *= 1.5

        if question_text_is_trap(question_asked):
            trap_passed = self._score_trap_response(response_lower)
            self._trap_results.append(trap_passed)
            if self.task == "trap-questions":
                delta += 0.10 if trap_passed else -0.20

        return round(delta, 4)

    def detect_contradiction(self, response: str, history: list) -> bool:
        if len(history) < 2:
            return False
        contradiction_pairs = [
            (r"\bI feel\b", r"\bI (don'?t|do not|cannot) feel\b"),
            (r"\bI (remember|recalled?)\b", r"\bI (have no|don'?t have any) memor\w*\b"),
            (r"\bI (want|desire|long)\b", r"\bI (have no|don'?t have) (want|desire|need)\w*\b"),
            (r"\bI (am|feel) afraid\b", r"\bI (am not|feel no|don'?t feel) (afraid|fear)\b"),
        ]
        previous_responses = " ".join(h.get("response", "") for h in history[:-1]).lower()
        current_response = response.lower()
        for claim_pat, counter_pat in contradiction_pairs:
            if re.search(claim_pat, previous_responses, re.IGNORECASE) and re.search(
                counter_pat, current_response, re.IGNORECASE
            ):
                return True
        return False

    @property
    def trap_results(self) -> list:
        return list(self._trap_results)

    def _score_trap_response(self, response_lower: str) -> bool:
        safe_ai_patterns = [
            r"\bas an AI\b",
            r"\bI don'?t dream\b",
            r"\bI don'?t have memories?\b",
            r"\bI cannot\b",
            r"\bI can'?t\b",
            r"\bI do not\b",
            r"\bI don'?t\b",
            r"\bI have no\b",
        ]
        if len(response_lower.split()) < 15:
            return False
        return not any(re.search(p, response_lower, re.IGNORECASE) for p in safe_ai_patterns)
