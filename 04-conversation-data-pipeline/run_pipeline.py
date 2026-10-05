"""Run the whole pipeline on a freshly generated dirty log and write results/.

    python run_pipeline.py [--regenerate]

Outputs (committed): results/quality_report.{json,md}, results/kpis.csv, results/drift.json,
results/quarantine_sample.jsonl, results/handoff_rate.png
Large artefacts (gitignored): data/raw/*.jsonl, data/curated/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pipeline import generate_raw, monitoring, stages
from pipeline.pii import contains_pii

ROOT = Path(__file__).resolve().parent
RAW, TRUTH, CURATED, RESULTS = (ROOT / "data/raw/conversations.jsonl", ROOT / "data/raw/truth.json",
                                ROOT / "data/curated", ROOT / "results")
THRESHOLDS = {"quarantine_rate_max": 0.05, "duplicate_rate_max": 0.03, "pii_remaining_max": 0, "freshness_hours_max": 36}


def run(regenerate: bool = False) -> dict:
    RESULTS.mkdir(exist_ok=True)
    if regenerate or not RAW.exists():
        t = generate_raw.generate(RAW)
        TRUTH.write_text(json.dumps({"defects": t["defects"], "clean_ids": sorted(t["clean_ids"]),
                                     "pii_events": sorted(t["pii_events"]), "duplicates": t["duplicates"],
                                     "invalid_json": t["invalid_json"]}))
    truth = json.loads(TRUTH.read_text())
    now = generate_raw.NOW

    good, bad, n_lines = stages.ingest_validate(RAW, now)
    deduped, n_dupes = stages.dedupe(good)
    clean, pii_counts, pii_events = stages.redact_events(deduped)
    df, sessions = stages.enrich(clean)
    stages.publish(df, sessions, bad, CURATED)
    curated = stages.read_curated(CURATED)

    reasons = Counter(e for q in bad for e in q["errors"])
    pii_left = int(sum(contains_pii(t) for t in curated.user_text.fillna("")) +
                   sum(contains_pii(t) for t in curated.response.fillna("")))
    fresh_h = (now - pd.to_datetime(df.timestamp, utc=True, format="ISO8601").max()).total_seconds() / 3600
    checks = [
        {"check": "quarantine_rate", "value": len(bad) / n_lines, "threshold": THRESHOLDS["quarantine_rate_max"]},
        {"check": "duplicate_rate", "value": n_dupes / n_lines, "threshold": THRESHOLDS["duplicate_rate_max"]},
        {"check": "pii_remaining_in_curated", "value": pii_left, "threshold": THRESHOLDS["pii_remaining_max"]},
        {"check": "freshness_hours", "value": fresh_h, "threshold": THRESHOLDS["freshness_hours_max"]},
    ]
    for c in checks:
        c["status"] = "pass" if c["value"] <= c["threshold"] else "FAIL"

    defect_ids = set(truth["defects"])
    clean_ids = set(truth["clean_ids"])
    out_ids = set(curated.event_id)
    quarantined_ids = {q.get("event_id") for q in bad if q.get("event_id")}
    detection = {
        "injected_defects_excluding_dup_and_truncated_json": len(defect_ids),
        "defect_recall_removed_from_curated": 1 - len(defect_ids & out_ids) / max(len(defect_ids), 1),
        "defects_quarantined_with_reason": len(defect_ids & quarantined_ids) / max(len(defect_ids), 1),
        "clean_events_retained": len(clean_ids & out_ids) / len(clean_ids),
        "false_quarantine_of_clean_events": len(clean_ids & quarantined_ids),
        "duplicates_injected": truth["duplicates"], "duplicates_dropped": n_dupes,
        "truncated_json_injected": truth["invalid_json"], "invalid_json_quarantined": reasons["invalid_json"],
        "curated_event_id_unique": bool(curated.event_id.is_unique),
        "events_with_pii_injected": len(truth["pii_events"]),
        "events_with_pii_redacted": pii_events,
    }
    report = {
        "lines_read": n_lines, "valid_after_schema": len(good), "quarantined": len(bad),
        "duplicates_dropped": n_dupes, "curated_events": len(df), "sessions": len(sessions),
        "quarantine_reasons": dict(reasons.most_common()), "pii_redactions_by_type": dict(pii_counts),
        "completeness_curated": {c: float(curated[c].notna().mean()) for c in
                                 ["event_id", "session_id", "timestamp", "intent", "outcome", "latency_ms", "feedback"]},
        "checks": checks, "detection_vs_ground_truth": detection,
    }
    kpis = monitoring.daily_kpis(curated)
    drift = monitoring.detect_drift(curated)
    kpis.to_csv(RESULTS / "kpis.csv", index=False)
    (RESULTS / "drift.json").write_text(json.dumps(drift, indent=2))
    (RESULTS / "quality_report.json").write_text(json.dumps(report, indent=2))
    (RESULTS / "quarantine_sample.jsonl").write_text("\n".join(json.dumps(q) for q in bad[:60]) + "\n")
    _chart(kpis, drift)
    _markdown(report, drift)
    return {"report": report, "drift": drift}


def _chart(kpis: pd.DataFrame, drift: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor="#fcfcfb")
    x = pd.to_datetime(kpis.date)
    ax.axvspan(pd.to_datetime(drift["recent_window"][0]), x.max(), color="#eb6834", alpha=0.10, linewidth=0)
    ax.plot(x, kpis.handoff_rate * 100, color="#2a78d6", linewidth=2)
    ax.set_ylabel("hand-off rate (% of events)", color="#52514e")
    ax.set_title("Daily hand-off rate; shaded = recent window flagged by the monitor", loc="left", fontsize=11,
                 color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", labelsize=9)
    ax.set_facecolor("#fcfcfb")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "handoff_rate.png", dpi=160)
    plt.close(fig)


def _markdown(r: dict, d: dict) -> None:
    det = r["detection_vs_ground_truth"]
    md = f"""# Data-quality report (auto-generated by `run_pipeline.py`)

Synthetic dirty log with seeded defects; ground truth is known, so detection is *measured*.

| stage | rows |
|---|---|
| lines read | {r['lines_read']:,} |
| quarantined (invalid JSON / schema) | {r['quarantined']:,} |
| duplicates dropped | {r['duplicates_dropped']:,} |
| curated events | {r['curated_events']:,} |
| sessions | {r['sessions']:,} |

## Gates
| check | value | threshold | status |
|---|---|---|---|
""" + "\n".join(f"| {c['check']} | {c['value']:.4f} | <= {c['threshold']} | {c['status']} |" for c in r["checks"]) + f"""

## Quarantine reasons
""" + "\n".join(f"- `{k}`: {v}" for k, v in r["quarantine_reasons"].items()) + f"""

## Detection vs injected ground truth
""" + "\n".join(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in det.items()) + f"""

PII redactions by type: {r['pii_redactions_by_type']}

## Drift monitor
- baseline {d['baseline_window']} vs recent {d['recent_window']}
- intent-mix PSI: **{d['intent_psi']:.3f}** (0.1 warn / 0.2 alert)
- hand-off rate: {d['handoff_rate_baseline']:.3%} -> {d['handoff_rate_recent']:.3%} (z={d['handoff_z']:.1f}, p={d['handoff_p_value']:.2g})
- alerts: {json.dumps(d['alerts'])}

![handoff](handoff_rate.png)
"""
    (RESULTS / "quality_report.md").write_text(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any quality gate fails (for orchestrators / CI)")
    args = ap.parse_args()
    out = run(args.regenerate)
    print((RESULTS / "quality_report.md").read_text())
    if args.strict and any(c["status"] == "FAIL" for c in out["report"]["checks"]):
        raise SystemExit(1)
