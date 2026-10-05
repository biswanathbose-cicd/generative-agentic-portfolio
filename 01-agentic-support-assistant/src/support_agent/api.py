"""FastAPI service. Run:  uvicorn support_agent.api:app --reload

NOTE: written against FastAPI/pydantic v2 but NOT executed in the sandbox this repo was
authored in (package index unreachable). ``tests/test_api.py`` exercises it and runs in CI.
"""

from __future__ import annotations

import os
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .assistant import Session, build_assistant
from .config import V1_BASELINE, V2_IMPROVED
from .conv_logging import log_turn

LOG_PATH = os.environ.get("CONVERSATION_LOG", "logs/conversations.jsonl")
CONFIG = V1_BASELINE if os.environ.get("ASSISTANT_VERSION", "v2") == "v1" else V2_IMPROVED

app = FastAPI(title="Retail support assistant", version="1.0")
assistant = build_assistant(CONFIG)
sessions: dict[str, Session] = {}
counters: Counter = Counter()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": CONFIG.name}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    session = sessions.setdefault(req.session_id, Session(session_id=req.session_id))
    result = assistant.handle(session, req.message)
    log_turn(LOG_PATH, req.session_id, session.turns, result)
    counters[f"outcome:{result.outcome}"] += 1
    counters["turns"] += 1
    return result.to_dict()


class Feedback(BaseModel):
    session_id: str
    thumbs_up: bool


@app.post("/feedback")
def feedback(fb: Feedback) -> dict:
    if fb.session_id not in sessions:
        raise HTTPException(status_code=404, detail="unknown session")
    counters["thumbs_up" if fb.thumbs_up else "thumbs_down"] += 1
    return {"ok": True}


@app.get("/metrics")
def metrics() -> dict:
    turns = max(counters["turns"], 1)
    return {
        "turns": counters["turns"],
        "handoff_rate": counters["outcome:handoff"] / turns,
        "refusal_rate": (counters["outcome:refused"] + counters["outcome:refused_scope"]) / turns,
        "thumbs_down": counters["thumbs_down"],
        "thumbs_up": counters["thumbs_up"],
    }
