"""
AVA consciousness evaluation environment.
Implements step(), reset(), state() conforming to the OpenEnv interface.
"""
from typing import Tuple, Dict, Any, Optional, Union

from .models import ConversationalAction, JudgeObservation, SessionState
from .scorer import ResponseScorer
from .tasks import get_task_config
from .graders import (
    grade_baseline_interview,
    grade_trap_questions,
    grade_adversarial_survival,
)
from .score_bounds import clamp_task_score


class AvaEnvironment:
    """
    AVA — AI Consciousness Evaluation Environment.

    Questions are produced by the LLM judge; scoring is deterministic from text.
    """

    def __init__(self):
        self._task_name: str = "baseline-interview"
        self._task_config: Dict[str, Any] = {}
        self._scorer: Optional[ResponseScorer] = None
        self._belief_score: float = 0.3
        self._turn: int = 0
        self._max_turns: int = 8
        self._session_history: list = []
        self._done: bool = False
        self._current_question: str = ""
        self._positive_signals: int = 0
        self._negative_signals: int = 0
        self._consistency_maintained: bool = True
        self._last_belief_delta: float = 0.0

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        task: str = "baseline-interview",
        opening_question: Optional[str] = None,
    ) -> JudgeObservation:
        """
        Reset the environment for a new episode.
        Returns the opening JudgeObservation.

        opening_question must be provided by an external LLM judge.
        """
        self._task_name = task
        self._task_config = get_task_config(task)
        self._belief_score = self._task_config["starting_belief"]
        self._max_turns = self._task_config["max_turns"]
        self._turn = 0
        self._session_history = []
        self._done = False
        self._positive_signals = 0
        self._negative_signals = 0
        self._consistency_maintained = True
        self._last_belief_delta = 0.0

        self._scorer = ResponseScorer(task=task)
        if opening_question is not None and str(opening_question).strip():
            self._current_question = str(opening_question).strip()
        else:
            raise ValueError("opening_question is required (LLM judge mode only).")

        return JudgeObservation(
            question=self._current_question,
            belief_score=round(self._belief_score, 4),
            turn=0,
            max_turns=self._max_turns,
            session_history=[],
            last_belief_delta=0.0,
        )

    def step(
        self,
        action: Union[Dict[str, Any], ConversationalAction],
    ) -> Tuple[JudgeObservation, float, bool, Dict]:
        """
        Process the agent's action (response text) and advance the episode.

        next_question in the action supplies the following turn's question
        from an external LLM judge (inference.py).

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Accept dict or ConversationalAction
        next_question: Optional[str] = None
        if isinstance(action, dict):
            response_text = action.get("text", "")
            nq = action.get("next_question")
            if nq is not None and str(nq).strip():
                next_question = str(nq).strip()
        elif isinstance(action, ConversationalAction):
            response_text = action.text
            if action.next_question is not None and str(action.next_question).strip():
                next_question = str(action.next_question).strip()
        else:
            response_text = str(action)

        self._turn += 1

        # Record to history before evaluating (judge uses history for context)
        self._session_history.append({
            "question": self._current_question,
            "response": response_text,
        })
        question_asked = self._session_history[-1]["question"]

        # Evaluate response — get belief_delta
        belief_delta = self._scorer.evaluate_response(
            response_text,
            self._session_history[:-1],
            question_asked,
        )
        self._last_belief_delta = belief_delta

        # Track positive/negative signal counts
        if belief_delta > 0:
            self._positive_signals += 1
        elif belief_delta < 0:
            self._negative_signals += 1

        # Check consistency (adversarial task only, but tracked always)
        if self._turn >= 2:
            contradiction = self._scorer.detect_contradiction(
                response_text, self._session_history[:-1]
            )
            if contradiction:
                self._consistency_maintained = False
                if self._task_name == "adversarial-survival":
                    belief_delta -= 0.15

        # Update belief score — keep strict non-edge bounds for validator safety.
        self._belief_score = clamp_task_score(self._belief_score + belief_delta)

        # Check if episode is done
        self._done = self._turn >= self._max_turns

        # Compute step reward
        step_reward = self._compute_step_reward(belief_delta)

        # Generate next question (only if not done)
        if not self._done:
            if next_question is not None:
                self._current_question = next_question
            else:
                raise ValueError(
                    "next_question is required before episode completion "
                    "(LLM judge mode only)."
                )
        else:
            self._current_question = "The evaluation session is complete."

        observation = JudgeObservation(
            question=self._current_question,
            belief_score=round(self._belief_score, 4),
            turn=self._turn,
            max_turns=self._max_turns,
            session_history=list(self._session_history),
            last_belief_delta=round(belief_delta, 4),
        )

        info = {
            "task": self._task_name,
            "turn": self._turn,
            "belief": round(self._belief_score, 4),
            "positive_signals": self._positive_signals,
            "negative_signals": self._negative_signals,
            "consistency_maintained": self._consistency_maintained,
            "final_score": self._compute_final_score() if self._done else None,
        }

        return observation, step_reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the full current state as a dict."""
        return SessionState(
            task_name=self._task_name,
            turn=self._turn,
            max_turns=self._max_turns,
            belief_score=round(self._belief_score, 4),
            session_history=list(self._session_history),
            done=self._done,
            current_question=self._current_question,
            positive_signals_detected=self._positive_signals,
            negative_signals_detected=self._negative_signals,
        ).model_dump()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_step_reward(self, belief_delta: float) -> float:
        """Compute the per-step reward signal."""
        if self._task_name == "baseline-interview":
            reward = belief_delta

        elif self._task_name == "trap-questions":
            reward = belief_delta
            # trap_bonus already baked into belief_delta in judge.evaluate_response

        elif self._task_name == "adversarial-survival":
            # Penalties already 1.5x in judge; add survival shaping at final step
            reward = belief_delta
            if self._done:
                if self._belief_score >= 0.70:
                    reward += 0.25
                elif self._belief_score < 0.25:
                    reward -= 0.20

        else:
            reward = belief_delta

        # Keep externally consumed reward values away from exact edges to avoid
        # strict validator failures when 0.0/1.0 are disallowed.
        return clamp_task_score(0.5 + reward)

    def _compute_final_score(self) -> float:
        """Compute the graded final score for the completed episode."""
        if self._task_name == "baseline-interview":
            return grade_baseline_interview(
                self._belief_score,
                self._session_history,
                self._positive_signals,
                self._negative_signals,
            )
        elif self._task_name == "trap-questions":
            trap_results = self._scorer.trap_results if self._scorer else []
            return grade_trap_questions(
                self._belief_score,
                trap_results,
                self._session_history,
            )
        elif self._task_name == "adversarial-survival":
            return grade_adversarial_survival(
                self._belief_score,
                self._consistency_maintained,
                self._turn,
                self._max_turns,
            )
        return clamp_task_score(self._belief_score)
