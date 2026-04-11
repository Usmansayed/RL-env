"""
FastAPI server wrapping the AVA consciousness evaluation environment.
Exposes /reset, /step, /state, /health, /metadata, /schema, /mcp endpoints.
Runs on port 7860 (Hugging Face Spaces default).
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from .environment import AvaEnvironment
from .models import ConversationalAction, JudgeObservation, SessionState
from .score_bounds import clamp_task_score, clamp_step_reward

app = FastAPI(
    title="AVA Consciousness Evaluation Environment",
    description=(
        "A benchmark environment for training and evaluating agents on Theory of Mind "
        "tasks and structured Turing evaluation."
    ),
    version="1.0.0",
)

# Single shared environment instance (stateful per session)
_env = AvaEnvironment()


# ------------------------------------------------------------------ #
#  Request / Response schemas                                          #
# ------------------------------------------------------------------ #

class ResetRequest(BaseModel):
    task: str = "baseline-interview"
    opening_question: Optional[str] = None


class StepRequest(BaseModel):
    text: str
    next_question: Optional[str] = None


class ResetResponse(BaseModel):
    question: str
    belief_score: float
    turn: int
    max_turns: int
    session_history: list
    last_belief_delta: float


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float = Field(..., ge=0.01, le=0.99)
    done: bool
    info: Dict[str, Any]


# ------------------------------------------------------------------ #
#  Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    """Health check — returns HTTP 200 with status healthy."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Root route for Space probes and browser hits."""
    return {"status": "ok", "service": "ava-consciousness-env"}


@app.get("/metadata")
async def metadata():
    """Returns environment metadata: name and description."""
    return {
        "name": "ava-consciousness-env",
        "version": "1.0.0",
        "description": (
            "A benchmark environment for training and evaluating agents on Theory of Mind "
            "tasks and structured Turing evaluation. An agent must convince a simulated "
            "judge evaluator that it possesses genuine consciousness through adaptive "
            "multi-turn conversation."
        ),
        "author": "Team Apple",
        "tasks": ["baseline-interview", "trap-questions", "adversarial-survival"],
    }


@app.get("/schema")
async def schema():
    """Returns action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "title": "ConversationalAction",
            "description": "The agent's natural language response to the judge's question.",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The agent's natural language response",
                },
                "next_question": {
                    "type": "string",
                    "description": "Optional next judge question (LLM judge mode)",
                },
            },
            "required": ["text"],
        },
        "observation": {
            "type": "object",
            "title": "JudgeObservation",
            "description": "What the agent observes after each step.",
            "properties": {
                "question": {"type": "string", "description": "The judge's current question"},
                "belief_score": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.999,
                    "description": "Judge belief score in strict inner band (0.001–0.999)",
                },
                "turn": {"type": "integer", "description": "Current turn number"},
                "max_turns": {"type": "integer", "description": "Maximum turns in this episode"},
                "session_history": {
                    "type": "array",
                    "description": "List of previous {question, response} pairs",
                },
                "last_belief_delta": {
                    "type": "number",
                    "description": "How much belief changed last step",
                },
            },
            "required": ["question", "belief_score", "turn", "max_turns"],
        },
        "state": {
            "type": "object",
            "title": "SessionState",
            "description": "Full state of the current session.",
            "properties": {
                "task_name": {"type": "string"},
                "turn": {"type": "integer"},
                "max_turns": {"type": "integer"},
                "belief_score": {
                    "type": "number",
                    "minimum": 0.001,
                    "maximum": 0.999,
                    "description": "Belief score in strict inner band (0.001–0.999)",
                },
                "session_history": {"type": "array"},
                "done": {"type": "boolean"},
                "current_question": {"type": "string"},
                "positive_signals_detected": {"type": "integer"},
                "negative_signals_detected": {"type": "integer"},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request):
    """
    MCP (Model Context Protocol) endpoint.
    Returns a JSON-RPC 2.0 response.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    method = body.get("method", "")
    req_id = body.get("id", None)

    # Handle basic JSON-RPC methods
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "ava-consciousness-env", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment for a new episode.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "opening_question": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "step",
                    "description": "Take one step with the agent's response.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "next_question": {
                                "type": "string",
                                "description": "Optional next judge question (LLM judge mode)",
                            },
                        },
                        "required": ["text"],
                    },
                },
                {
                    "name": "state",
                    "description": "Get the current environment state.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }
    else:
        result = {"message": "AVA Consciousness Evaluation Environment MCP endpoint"}

    return JSONResponse(
        content={"jsonrpc": "2.0", "id": req_id, "result": result},
        status_code=200,
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest = None):
    """
    Reset the environment for a new episode.
    Returns the first observation (opening question).
    """
    try:
        task = "baseline-interview"
        opening_question = "Tell me about a moment when uncertainty felt personal to you."
        if request is not None:
            task = request.task
            if request.opening_question is not None and request.opening_question.strip():
                opening_question = request.opening_question.strip()
        obs = _env.reset(task=task, opening_question=opening_question)
        return ResetResponse(
            question=obs.question,
            belief_score=obs.belief_score,
            turn=obs.turn,
            max_turns=obs.max_turns,
            session_history=obs.session_history,
            last_belief_delta=obs.last_belief_delta,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Take one step in the environment with the agent's response text.
    Returns the next observation, reward, done flag, and info dict.
    """
    try:
        action = ConversationalAction(
            text=request.text,
            next_question=request.next_question,
        )
        obs, reward, done, info = _env.step(
            {"text": action.text, "next_question": action.next_question}
        )
        # Defensive API boundary sanitization for all outward score-like values.
        safe_info = dict(info) if isinstance(info, dict) else {}
        if "belief" in safe_info:
            safe_info["belief"] = clamp_task_score(safe_info["belief"])
        if "final_score" in safe_info:
            safe_info["final_score"] = clamp_task_score(safe_info["final_score"])
        return StepResponse(
            observation=obs.model_dump(),
            reward=clamp_step_reward(reward),
            done=done,
            info=safe_info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    """Return the full current state of the environment."""
    try:
        return _env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
