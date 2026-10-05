"""Baseline, linear and gradient-boosted forecasters + metrics + rolling-origin evaluation."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import CATEGORICAL, FEATURES, H, NUMERIC


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / y.sum())


def bias(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p - y).sum() / y.sum())


class SeasonalNaive:
    """Same weekday last week: with H = 7 that is exactly ``lag_7``."""
    name = "seasonal_naive"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["lag_7"].to_numpy(float)


class LogTargetModel:
    def __init__(self, name, estimator):
        self.name, self.est = name, estimator

    def fit(self, X, y):
        self.est.fit(X[FEATURES], np.log1p(y))
        return self

    def predict(self, X):
        return np.expm1(self.est.predict(X[FEATURES])).clip(min=0)


LOG_COLS = [c for c in NUMERIC if c.startswith(("lag_", "roll_"))]
PLAIN_COLS = [c for c in NUMERIC if c not in LOG_COLS]


def make_ridge(alpha: float = 3.0) -> LogTargetModel:
    """Linear baseline on a log target; history features are log1p-scaled so the model is
    additive in log space (a fair baseline for multiplicative demand, not a straw man)."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), LOG_COLS),
        ("num", StandardScaler(), PLAIN_COLS)])
    return LogTargetModel("ridge", make_pipeline(pre, Ridge(alpha=alpha)))


def make_hgb(name="hist_gradient_boosting", **params) -> LogTargetModel:
    mask = [c in CATEGORICAL for c in FEATURES]
    defaults = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                    categorical_features=mask, random_state=0)
    defaults.update(params)
    return LogTargetModel(name, HistGradientBoostingRegressor(**defaults))


def make_quantile(q: float, **params) -> LogTargetModel:
    mask = [c in CATEGORICAL for c in FEATURES]
    return LogTargetModel(f"q{q}", HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=200, learning_rate=0.08, categorical_features=mask,
        random_state=0, **params))


def split_windows(dates: pd.Series, n_folds: int = 4, window: int = 28):
    """Rolling-origin folds: each test window follows a training set that ends H days earlier
    (the last day whose actuals are known when the first test-day forecast is issued)."""
    last = dates.max()
    for k in range(n_folds):
        test_end = last - pd.Timedelta(days=(n_folds - 1 - k) * window)
        test_start = test_end - pd.Timedelta(days=window - 1)
        yield k, test_start, test_end, test_start - pd.Timedelta(days=H + 1)


def tune_hgb(feat: pd.DataFrame, first_test_start: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """Small grid search with a time-ordered validation block that ends before every test fold."""
    val_end = first_test_start - pd.Timedelta(days=H + 1)
    val_start = val_end - pd.Timedelta(days=27)
    train = feat[feat.date <= val_start - pd.Timedelta(days=H + 1)]
    val = feat[(feat.date >= val_start) & (feat.date <= val_end)]
    rows = []
    for lr, leaves, l2 in itertools.product([0.05, 0.1], [15, 31], [0.0, 1.0]):
        m = make_hgb(learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=l2).fit(train, train.units)
        rows.append({"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2,
                     "val_wape": wape(val.units, m.predict(val))})
    grid = pd.DataFrame(rows).sort_values("val_wape").reset_index(drop=True)
    best = grid.iloc[0]
    return {"learning_rate": float(best.learning_rate), "max_leaf_nodes": int(best.max_leaf_nodes),
            "l2_regularization": float(best.l2_regularization)}, grid


def rolling_evaluation(feat: pd.DataFrame, hgb_params: dict, n_folds: int = 4, window: int = 28):
    rows, preds = [], []
    for k, ts, te, train_end in split_windows(feat.date, n_folds, window):
        train, test = feat[feat.date <= train_end], feat[(feat.date >= ts) & (feat.date <= te)]
        models = [SeasonalNaive(), make_ridge(), make_hgb(**hgb_params)]
        out = test[["date", "store", "dept", "units"]].copy()
        out["fold"] = k
        for m in models:
            m.fit(train, train.units)
            p = m.predict(test)
            out[m.name] = p
            rows.append({"fold": k, "model": m.name, "wape": wape(test.units, p), "bias": bias(test.units, p),
                         "mae": float(np.abs(test.units - p).mean())})
        preds.append(out)
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)
