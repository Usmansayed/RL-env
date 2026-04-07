from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ConversationalAction(BaseModel):
    """The agent's response to the judge's question."""
    text: str = Field(..., description="The agent's natural language response")
    next_question: Optional[str] = Field(
        default=None,
        description=(
            "Optional next judge question for the following turn. "
            "Required before episode completion when using the runtime environment."
        ),
    )


class JudgeObservation(BaseModel):
    """What the agent observes after each step."""
    question: str = Field(..., description="The judge's current question")
    belief_score: float = Field(..., ge=0.0, le=1.0,
                                description="Judge's belief score (0=not conscious, 1=conscious)")
    turn: int = Field(..., description="Current turn number (1-indexed)")
    max_turns: int = Field(..., description="Maximum turns in this episode")
    session_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of previous {question, response} pairs"
    )
    last_belief_delta: float = Field(
        default=0.0,
        description="How much belief changed last step"
    )


class SessionState(BaseModel):
    """Full state of the current session."""
    task_name: str
    turn: int
    max_turns: int
    belief_score: float
    session_history: List[Dict[str, str]]
    done: bool
    current_question: str
    positive_signals_detected: int
    negative_signals_detected: int
