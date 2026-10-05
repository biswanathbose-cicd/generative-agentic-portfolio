"""Synthetic daily sales with known structure (weekly + annual seasonality, holidays,
week-long promotions, trend, count noise). Everything is generated; no real retail data.

``true_rate`` is the Poisson rate that generated each observation. It is NEVER used as a feature;
it exists only to compute an oracle noise-floor WAPE for context."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

DEPTS = ["grocery", "electronics", "apparel", "home"]
DOW = {  # Mon..Sun multipliers per department
    "grocery": [0.95, 0.9, 0.92, 0.97, 1.08, 1.25, 1.12],
    "electronics": [0.85, 0.85, 0.88, 0.95, 1.1, 1.3, 1.2],
    "apparel": [0.8, 0.82, 0.88, 0.95, 1.1, 1.35, 1.25],
    "home": [0.9, 0.9, 0.92, 0.98, 1.05, 1.25, 1.15],
}


def holiday_dates(year: int) -> list[date]:
    thanksgiving = [d for d in (date(year, 11, 1) + timedelta(days=i) for i in range(30))
                    if d.month == 11 and d.weekday() == 3][3]
    return [date(year, 7, 4), thanksgiving, date(year, 12, 24)]


def holiday_lift(d: date, dept: str) -> float:
    lift = 1.0
    for h in holiday_dates(d.year):
        gap = (h - d).days
        if 0 <= gap <= 3:
            strength = {"grocery": 0.5, "electronics": 0.7, "apparel": 0.4, "home": 0.3}[dept]
            lift *= 1 + strength * (1 - gap / 4)
    return lift


def generate_sales(n_stores: int = 6, days: int = 1095, start: str = "2023-01-02", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()
    t = np.arange(days)
    frames = []
    for store in range(n_stores):
        store_scale = rng.uniform(0.7, 1.5)
        for dept in DEPTS:
            base = rng.uniform(40, 200) * store_scale * {"grocery": 1.6, "electronics": 0.6,
                                                         "apparel": 0.9, "home": 0.8}[dept]
            amp = rng.uniform(0.05, 0.25)
            phase = rng.uniform(0, 2 * np.pi)
            annual = 1 + amp * np.sin(2 * np.pi * doy / 365.25 + phase)
            dow = np.array(DOW[dept])[dates.dayofweek.to_numpy()]
            hol = np.array([holiday_lift(d.date(), dept) for d in dates])
            trend = 1 + rng.uniform(0.0, 0.0004) * t
            promo_weeks = rng.random(days // 7 + 2) < 0.15
            promo = np.repeat(promo_weeks, 7)[:days].astype(int)
            promo_lift = 1 + promo * rng.uniform(0.2, 0.45)
            price_base = {"grocery": 4.0, "electronics": 60.0, "apparel": 25.0, "home": 30.0}[dept]
            price = np.round(price_base * (1 - 0.2 * promo) * rng.uniform(0.97, 1.03, days), 2)
            mean = base * dow * annual * hol * trend * promo_lift * rng.lognormal(0, 0.08, days)
            units = rng.poisson(mean)
            frames.append(pd.DataFrame({"date": dates, "store": store, "dept": dept, "units": units,
                                        "price": price, "promo": promo, "true_rate": mean}))
    return pd.concat(frames, ignore_index=True)
