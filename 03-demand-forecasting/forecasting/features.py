"""Leak-free feature engineering for a fixed forecast horizon.

A row for day ``t`` is built as if the forecast were issued on day ``t - H``:
history features use only ``units`` observed on or before ``t - H``. Promotions and prices
are *planned in advance*, so they are legitimately known at forecast time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import holiday_dates

H = 7
LAGS = (H, H + 7, H + 14, H + 21)
ROLL_WINDOWS = (7, 28)
CATEGORICAL = ["store", "dept", "dow"]
NUMERIC = [f"lag_{l}" for l in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [
    "roll_std_28", "promo", "price", "doy_sin", "doy_cos", "days_to_holiday", "trend_idx"]
FEATURES = CATEGORICAL + NUMERIC


def _days_to_next_holiday(dates: pd.Series) -> np.ndarray:
    out = []
    for d in dates.dt.date:
        gaps = [(h - d).days for y in (d.year, d.year + 1) for h in holiday_dates(y) if (h - d).days >= 0]
        out.append(min(min(gaps), 30))
    return np.array(out)


def make_features(df: pd.DataFrame, h: int = H) -> pd.DataFrame:
    df = df.sort_values(["store", "dept", "date"]).reset_index(drop=True).copy()
    g = df.groupby(["store", "dept"], sort=False)["units"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(h)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = shifted.groupby([df.store, df.dept], sort=False).transform(
            lambda s, w=w: s.rolling(w).mean())
    df["roll_std_28"] = shifted.groupby([df.store, df.dept], sort=False).transform(lambda s: s.rolling(28).std())
    df["dow"] = df.date.dt.dayofweek
    doy = df.date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["days_to_holiday"] = _days_to_next_holiday(df.date)
    df["trend_idx"] = (df.date - df.date.min()).dt.days
    df["dept"] = pd.Categorical(df["dept"], categories=sorted(df["dept"].unique())).codes
    df = df.dropna(subset=NUMERIC).reset_index(drop=True)
    return df
