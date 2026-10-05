"""Append-only JSONL conversation log. The schema is the contract consumed by
``04-conversation-data-pipeline``."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .assistant import Result


def log_turn(path: str | Path, session_id: str, turn: int, result: Result, ts: datetime | None = None) -> dict:
    record = {
        "event_id": str(uuid.uuid4()),
        "session_id": session_id,
        "turn": turn,
        "timestamp": (ts or datetime.now(timezone.utc)).isoformat(),
        "user_text": result.logged_input,  # already PII-redacted when input guardrails are on
        "intent": result.intent,
        "outcome": result.outcome,
        "tools": [t["tool"] for t in result.trace],
        "response": result.response,
        "latency_ms": round(result.latency_ms, 3),
        "guardrail_flags": result.guardrail_flags,
        "feedback": None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
