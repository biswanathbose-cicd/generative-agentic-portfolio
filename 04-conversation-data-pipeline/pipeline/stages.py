"""Pipeline stages. Each stage is a small pure function so it can be unit-tested and, later,
wrapped as an Airflow/Prefect task without changes."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from .pii import redact
from .schema import parse_ts, validate_event


def ingest_validate(raw_path: str | Path, now: datetime) -> tuple[list[dict], list[dict], int]:
    """Read JSONL. Returns (valid events, quarantined records, lines read)."""
    good, bad, n = [], [], 0
    with open(raw_path) as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                bad.append({"line": i, "errors": ["invalid_json"], "raw": line[:300]})
                continue
            if not isinstance(ev, dict):
                bad.append({"line": i, "errors": ["not_an_object"], "raw": line[:300]})
                continue
            errs = validate_event(ev, now)
            if errs:
                bad.append({"line": i, "errors": errs, "raw": line[:300], "event_id": ev.get("event_id")})
            else:
                good.append(ev)
    return good, bad, n


def dedupe(events: list[dict]) -> tuple[list[dict], int]:
    """Keep the earliest copy of each event_id (at-least-once delivery produces repeats)."""
    seen: dict[str, dict] = {}
    for ev in sorted(events, key=lambda e: parse_ts(e["timestamp"])):
        seen.setdefault(ev["event_id"], ev)
    return list(seen.values()), len(events) - len(seen)


def redact_events(events: list[dict]) -> tuple[list[dict], Counter, int]:
    totals: Counter = Counter()
    touched = 0
    out = []
    for ev in events:
        ev = dict(ev)
        hit = False
        for field in ("user_text", "response"):
            ev[field], counts = redact(ev[field])
            if counts:
                hit = True
                totals.update(counts)
        touched += hit
        out.append(ev)
    return out, totals, touched


def enrich(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(events)
    df["ts"] = pd.to_datetime(df.timestamp, utc=True, format="ISO8601")
    df["date"] = df.ts.dt.strftime("%Y-%m-%d")
    df["hour"] = df.ts.dt.hour
    df["escalated"] = df.outcome.eq("handoff")
    df["refused"] = df.outcome.isin(["refused", "refused_scope"])
    df["n_tools"] = df.tools.map(len)
    df["user_text_len"] = df.user_text.str.len()
    df["tools"] = df.tools.map("|".join)
    df["guardrail_flags"] = df.guardrail_flags.map(lambda x: "|".join(x) if isinstance(x, list) else "")
    df = df.sort_values(["session_id", "turn", "ts"]).reset_index(drop=True)
    sessions = df.groupby("session_id").agg(
        started_at=("ts", "min"), ended_at=("ts", "max"), n_turns=("turn", "max"), n_events=("event_id", "count"),
        escalated=("escalated", "max"), first_intent=("intent", "first"), last_outcome=("outcome", "last"),
    ).reset_index()
    sessions["duration_s"] = (sessions.ended_at - sessions.started_at).dt.total_seconds()
    return df.drop(columns=["ts"]), sessions


def publish(df: pd.DataFrame, sessions: pd.DataFrame, quarantine: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    for date, g in df.groupby("date"):
        part = out / "events" / f"date={date}"
        part.mkdir(parents=True, exist_ok=True)
        g.drop(columns=["date"]).to_csv(part / "part-0.csv", index=False)
    sessions.to_csv(out / "sessions.csv", index=False)
    with open(out / "quarantine.jsonl", "w") as f:
        for q in quarantine:
            f.write(json.dumps(q) + "\n")


def read_curated(out_dir: str | Path) -> pd.DataFrame:
    frames = []
    for p in sorted(Path(out_dir, "events").glob("date=*/part-0.csv")):
        d = pd.read_csv(p, keep_default_na=False, na_values=[""])
        d["date"] = p.parent.name.split("=", 1)[1]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
