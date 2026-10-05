"""Production monitoring on curated conversations: daily KPIs + drift detection (PSI and a
two-proportion test on the hand-off rate)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date")
    k = pd.DataFrame({
        "events": g.size(),
        "sessions": g.session_id.nunique(),
        "handoff_rate": g.escalated.mean(),
        "refusal_rate": g.refused.mean(),
        "clarify_rate": g.outcome.apply(lambda s: (s == "clarify").mean()),
        "latency_ms_p95": g.latency_ms.quantile(0.95),
        "thumbs_down_share": g.feedback.apply(lambda s: (s == "down").sum() / max((s.notna()).sum(), 1)),
    })
    return k.reset_index()


def psi(baseline: pd.Series, recent: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(baseline.unique()) | set(recent.unique()))
    b = baseline.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    r = recent.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    return float(((r - b) * np.log(r / b)).sum())


def detect_drift(df: pd.DataFrame, baseline_days: int = 14, recent_days: int = 7) -> dict:
    dates = sorted(df.date.unique())
    base, rec = df[df.date.isin(dates[:baseline_days])], df[df.date.isin(dates[-recent_days:])]
    p_intent = psi(base.intent, rec.intent)
    x1, n1, x2, n2 = int(base.escalated.sum()), len(base), int(rec.escalated.sum()), len(rec)
    p1, p2, pool = x1 / n1, x2 / n2, (x1 + x2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se else 0.0
    p_val = float(2 * (1 - stats.norm.cdf(abs(z))))
    alerts = []
    if p_intent > 0.2:
        alerts.append({"severity": "alert", "signal": "intent_mix_psi", "value": p_intent, "threshold": 0.2})
    elif p_intent > 0.1:
        alerts.append({"severity": "warn", "signal": "intent_mix_psi", "value": p_intent, "threshold": 0.1})
    if p_val < 0.001 and p1 > 0 and (p2 / p1 - 1) > 0.25:
        alerts.append({"severity": "alert", "signal": "handoff_rate_increase", "value": p2 / p1 - 1,
                       "threshold": 0.25, "p_value": p_val})
    return {"baseline_window": [dates[0], dates[min(baseline_days, len(dates)) - 1]],
            "recent_window": [dates[-recent_days], dates[-1]], "intent_psi": p_intent,
            "handoff_rate_baseline": p1, "handoff_rate_recent": p2, "handoff_z": z, "handoff_p_value": p_val,
            "alerts": alerts}
