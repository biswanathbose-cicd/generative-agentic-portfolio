"""End-to-end experiment: data -> features -> tuning -> rolling-origin evaluation ->
interval coverage -> permutation importance -> business-impact memo.

    python run_experiment.py        # ~20 s, writes results/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

from forecasting.data import generate_sales
from forecasting.features import FEATURES, make_features
from forecasting.models import (SeasonalNaive, make_hgb, make_quantile, make_ridge, rolling_evaluation,
                                split_windows, tune_hgb, wape)

RESULTS = Path(__file__).resolve().parent / "results"
INK, MUTED = "#0b0b0b", "#52514e"
DEPT_NAMES = ["apparel", "electronics", "grocery", "home"]  # order of pd.Categorical codes (sorted)
# ASSUMED business parameters for the illustrative impact estimate (not real retailer data)
UNIT_COST = {"apparel": 18.0, "electronics": 45.0, "grocery": 3.0, "home": 22.0}
HOLDING_RATE = 0.20
SERVICE_LEVEL = 0.95


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_facecolor("#fcfcfb")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    raw = generate_sales()
    floor = wape(raw.units, raw.true_rate)  # oracle: knows the generating Poisson rate
    feat = make_features(raw)
    assert "true_rate" not in FEATURES

    first_test = next(split_windows(feat.date))[1]
    best, grid = tune_hgb(feat, first_test)
    fold_df, preds = rolling_evaluation(feat, best)
    fold_df.to_csv(RESULTS / "fold_metrics.csv", index=False)
    grid.to_csv(RESULTS / "tuning_grid.csv", index=False)
    summary = fold_df.groupby("model")[["wape", "bias", "mae"]].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(c) for c in summary.columns]

    # ---- prediction intervals (10th-90th percentile) pooled over folds
    covered, widths, n = 0, [], 0
    last_band = None
    for k, ts, te, train_end in split_windows(feat.date):
        train, test = feat[feat.date <= train_end], feat[(feat.date >= ts) & (feat.date <= te)]
        lo = make_quantile(0.1).fit(train, train.units).predict(test)
        hi = make_quantile(0.9).fit(train, train.units).predict(test)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        covered += int(((test.units >= lo) & (test.units <= hi)).sum())
        n += len(test)
        widths.append(float(((hi - lo) / test.units.clip(lower=1)).median()))
        last_band = (test.assign(lo=lo, hi=hi), k)
    coverage = covered / n

    # ---- permutation importance on the last fold
    k, ts, te, train_end = list(split_windows(feat.date))[-1]
    train, test = feat[feat.date <= train_end], feat[(feat.date >= ts) & (feat.date <= te)]
    model = make_hgb(**best).fit(train, train.units)
    imp = permutation_importance(model, test[FEATURES], test.units, n_repeats=5, random_state=0,
                                 scoring=lambda est, X, y: -float(np.abs(y - est.predict(X)).mean()))
    importance = (pd.DataFrame({"feature": FEATURES, "mae_increase_when_shuffled": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("mae_increase_when_shuffled", ascending=False).reset_index(drop=True))
    importance.to_csv(RESULTS / "permutation_importance.csv", index=False)

    # ---- illustrative business impact (ASSUMED costs): safety stock implied by forecast error
    pr = preds.copy()
    z = stats.norm.ppf(SERVICE_LEVEL)
    rows = []
    for (store, dept), g in pr.groupby(["store", "dept"]):
        row = {"store": store, "dept": dept}
        for m in ("seasonal_naive", "hist_gradient_boosting"):
            row[m] = z * float((g.units - g[m]).std())
        rows.append(row)
    ss = pd.DataFrame(rows)
    ss["unit_cost"] = ss.dept.map(lambda c: UNIT_COST[DEPT_NAMES[int(c)]])
    cost_naive = float((ss.seasonal_naive * ss.unit_cost).sum() * HOLDING_RATE)
    cost_model = float((ss.hist_gradient_boosting * ss.unit_cost).sum() * HOLDING_RATE)

    metrics = {
        "n_series": int(raw.groupby(["store", "dept"]).ngroups), "n_days": int(raw.date.nunique()),
        "horizon_days": 7, "best_hgb_params": best, "oracle_noise_floor_wape": floor,
        "rolling_origin_wape_mean": {m: float(summary.loc[m, "wape_mean"]) for m in summary.index},
        "rolling_origin_wape_std": {m: float(summary.loc[m, "wape_std"]) for m in summary.index},
        "rolling_origin_bias_mean": {m: float(summary.loc[m, "bias_mean"]) for m in summary.index},
        "interval_80_coverage": coverage, "interval_median_relative_width": float(np.median(widths)),
        "wape_reduction_vs_seasonal_naive_pct": float(
            (1 - summary.loc["hist_gradient_boosting", "wape_mean"] / summary.loc["seasonal_naive", "wape_mean"]) * 100),
        "assumed_annual_holding_cost_naive_usd": cost_naive,
        "assumed_annual_holding_cost_model_usd": cost_model,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- charts
    fig, ax = plt.subplots(figsize=(6.5, 3.8), facecolor="#fcfcfb")
    names = ["seasonal_naive", "ridge", "hist_gradient_boosting"]
    vals = [summary.loc[m, "wape_mean"] * 100 for m in names]
    bars = ax.bar(range(3), vals, 0.55, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.axhline(floor * 100, color="#eb6834", linewidth=1.5, linestyle="--")
    ax.text(2.45, floor * 100 + 0.5, f"noise floor {floor * 100:.1f}%", ha="right", fontsize=9, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["seasonal naive", "ridge", "gradient boosting"])
    ax.set_ylabel("WAPE, 7-day-ahead (%)", color=MUTED)
    ax.set_title("Rolling-origin WAPE (4 folds x 28 days)", loc="left", fontsize=11, color=INK)
    ax.set_ylim(0, max(vals) * 1.2)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "wape_by_model.png", dpi=160)
    plt.close(fig)

    band, _ = last_band
    sd = band[(band.store == 0) & (band.dept == band.dept.iloc[0])].sort_values("date")
    pred_hgb = preds[(preds.fold == 3) & (preds.store == 0) & (preds.dept == sd.dept.iloc[0])].sort_values("date")
    fig, ax = plt.subplots(figsize=(7.5, 3.8), facecolor="#fcfcfb")
    ax.fill_between(sd.date, sd.lo, sd.hi, color="#2a78d6", alpha=0.18, linewidth=0, label="80% interval")
    ax.plot(sd.date, sd.units, color=INK, linewidth=1.6, label="actual")
    ax.plot(pred_hgb.date, pred_hgb.hist_gradient_boosting, color="#2a78d6", linewidth=2, label="forecast (7 days ahead)")
    ax.set_ylabel("units / day", color=MUTED)
    ax.set_title("One store-department series, final 28-day test window", loc="left", fontsize=11, color=INK, pad=24)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    style(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "forecast_example.png", dpi=160)
    plt.close(fig)

    # ---- memo (generated)
    w = metrics["rolling_origin_wape_mean"]
    memo = f"""# Business-impact memo (auto-generated)

**Data:** synthetic - {metrics['n_series']} store x department series, {metrics['n_days']} days. Findings here validate the *method*, not any retailer's reality.

## What was built
A 7-day-ahead unit-demand forecaster. Features use only information available 7 days before the target day
(lags >= 7, shifted rolling stats) plus planned promotions/prices. Evaluation is rolling-origin: 4 folds x 28 days,
each trained only on data known when the first forecast of the fold would have been issued. Hyper-parameters were tuned
on a time-ordered validation block that ends before the first test fold.

## Results (WAPE, lower is better)
| model | WAPE mean | WAPE std across folds | bias |
|---|---|---|---|
| seasonal naive (same weekday last week) | {w['seasonal_naive']:.3f} | {metrics['rolling_origin_wape_std']['seasonal_naive']:.3f} | {metrics['rolling_origin_bias_mean']['seasonal_naive']:+.3f} |
| ridge (log target, log lags) | {w['ridge']:.3f} | {metrics['rolling_origin_wape_std']['ridge']:.3f} | {metrics['rolling_origin_bias_mean']['ridge']:+.3f} |
| gradient boosting | {w['hist_gradient_boosting']:.3f} | {metrics['rolling_origin_wape_std']['hist_gradient_boosting']:.3f} | {metrics['rolling_origin_bias_mean']['hist_gradient_boosting']:+.3f} |
| *oracle that knows the true Poisson rate* | *{floor:.3f}* | - | - |

Gradient boosting cuts WAPE by **{metrics['wape_reduction_vs_seasonal_naive_pct']:.0f}%** versus seasonal naive. The oracle row is the
irreducible count-noise floor of this synthetic data: on real data no such floor is knowable, so an equivalent
sanity check would be a held-out comparison against the planners' current forecast.

The 10th-90th percentile interval covers **{coverage:.1%}** of actuals (nominal 80%), median width {metrics['interval_median_relative_width']:.2f}x actual units.

## Illustrative cost translation (ASSUMED parameters - replace with real ones)
Safety stock proxy = z({SERVICE_LEVEL:.0%}) x std of 7-day forecast error per series. Unit costs (USD): {UNIT_COST}. Annual holding rate {HOLDING_RATE:.0%}.

| | annual holding cost of safety stock (USD, 24 series) |
|---|---|
| seasonal naive | {cost_naive:,.0f} |
| gradient boosting | {cost_model:,.0f} |
| difference | {cost_naive - cost_model:,.0f} ({(1 - cost_model / cost_naive) * 100:.0f}% lower) |

This ignores lead-time aggregation, demand correlation across series and stock-out cost, so treat it as an order-of-magnitude
argument for *why forecast error matters*, not as a savings forecast.

## Top drivers (permutation importance, final fold; MAE increase when shuffled)
{chr(10).join(f"- `{r.feature}`: +{r.mae_increase_when_shuffled:.2f}" for r in importance.head(6).itertuples())}

## Risks / next steps
- Promotions are known in advance here; in practice promo calendars change late - evaluate with noisy promo flags.
- No new-item or cold-start handling; no hierarchy reconciliation (store -> region -> total).
- Monitor WAPE and bias per department weekly; alert on drift of `lag`/`roll` feature distributions.
"""
    (RESULTS / "business_memo.md").write_text(memo)
    print(json.dumps(metrics, indent=2))
    print(importance.head(8).to_string())


if __name__ == "__main__":
    main()
