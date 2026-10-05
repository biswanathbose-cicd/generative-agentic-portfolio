"""Declarative validation rules for one conversation-log event.

The schema mirrors what ``01-agentic-support-assistant`` writes via ``conv_logging.log_turn``.
``validate_event`` returns a list of error codes (empty list == valid)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

INTENTS = {"order_status", "return_request", "product_search", "human_handoff", "out_of_scope", "blocked"}
OUTCOMES = {"answered", "clarify", "handoff", "refused", "refused_scope", "error"}
REQUIRED = ["event_id", "session_id", "turn", "timestamp", "user_text", "intent", "outcome",
            "tools", "response", "latency_ms"]
MAX_LATENCY_MS = 60_000
MAX_AGE = timedelta(days=730)


def parse_ts(value) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def validate_event(e: dict, now: datetime) -> list[str]:
    errors: list[str] = []
    for f in REQUIRED:
        if f not in e or e[f] is None:
            errors.append(f"missing:{f}")
    if errors:
        return errors
    for f in ("event_id", "session_id", "user_text", "response"):
        if not isinstance(e[f], str) or (f in ("event_id", "session_id") and not e[f].strip()):
            errors.append(f"type:{f}")
    if not isinstance(e["turn"], int) or isinstance(e["turn"], bool):
        errors.append("type:turn")
    elif e["turn"] < 1:
        errors.append("range:turn")
    if not isinstance(e["tools"], list) or not all(isinstance(t, str) for t in e["tools"]):
        errors.append("type:tools")
    lat = e["latency_ms"]
    if isinstance(lat, bool) or not isinstance(lat, (int, float)):
        errors.append("type:latency_ms")
    elif not 0 <= lat <= MAX_LATENCY_MS:
        errors.append("range:latency_ms")
    if e["intent"] not in INTENTS:
        errors.append("enum:intent")
    if e["outcome"] not in OUTCOMES:
        errors.append("enum:outcome")
    ts = parse_ts(e["timestamp"])
    if ts is None:
        errors.append("bad_timestamp")
    elif ts > now + timedelta(minutes=5):
        errors.append("future_timestamp")
    elif now - ts > MAX_AGE:
        errors.append("stale_timestamp")
    fb = e.get("feedback")
    if fb not in (None, "up", "down"):
        errors.append("enum:feedback")
    return errors
