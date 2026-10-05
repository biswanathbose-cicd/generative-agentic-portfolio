import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.data import generate_sales, holiday_dates  # noqa: E402
from forecasting.features import FEATURES, H, make_features  # noqa: E402
from forecasting.models import (SeasonalNaive, bias, make_hgb, make_ridge, split_windows,  # noqa: E402
                                wape)

RAW = generate_sales(n_stores=2, days=500, seed=3)
FEAT = make_features(RAW)


class DataTests(unittest.TestCase):
    def test_deterministic(self):
        pd.testing.assert_frame_equal(generate_sales(n_stores=1, days=60, seed=1),
                                      generate_sales(n_stores=1, days=60, seed=1))

    def test_non_negative_integers(self):
        self.assertTrue((RAW.units >= 0).all())
        self.assertTrue(np.issubdtype(RAW.units.dtype, np.integer))

    def test_thanksgiving_is_fourth_thursday(self):
        t = holiday_dates(2025)[1]
        self.assertEqual((t.month, t.day, t.weekday()), (11, 27, 3))

    def test_oracle_rate_is_not_a_feature(self):
        self.assertNotIn("true_rate", FEATURES)
        self.assertNotIn("units", FEATURES)


class LeakageTests(unittest.TestCase):
    def test_features_for_day_t_ignore_actuals_after_t_minus_h(self):
        """Corrupt every actual from day D onward: features for rows with t < D + H must not move."""
        cutoff = RAW.date.min() + pd.Timedelta(days=300)
        corrupted = RAW.copy()
        corrupted.loc[corrupted.date >= cutoff, "units"] = 10 ** 6
        a, b = make_features(RAW), make_features(corrupted)
        key = ["store", "dept", "date"]
        m = a.merge(b, on=key, suffixes=("_a", "_b"))
        safe = m[m.date < cutoff + pd.Timedelta(days=H)]
        self.assertGreater(len(safe), 1000)
        for f in [c for c in FEATURES if c not in ("store", "dept", "dow")]:
            self.assertTrue(np.allclose(safe[f + "_a"], safe[f + "_b"]), f)

    def test_folds_do_not_overlap_and_train_ends_before_forecast_origin(self):
        folds = list(split_windows(FEAT.date, n_folds=3, window=14))
        for (_, ts, te, train_end), (_, ts2, _, _) in zip(folds, folds[1:]):
            self.assertLess(te, ts2)
        for _, ts, te, train_end in folds:
            self.assertLessEqual(train_end, ts - pd.Timedelta(days=H))


class MetricTests(unittest.TestCase):
    def test_wape_and_bias(self):
        self.assertAlmostEqual(wape([10, 10], [8, 12]), 0.2)
        self.assertAlmostEqual(bias([10, 10], [12, 12]), 0.2)


class ModelTests(unittest.TestCase):
    def setUp(self):
        cut = FEAT.date.max() - pd.Timedelta(days=28)
        self.train = FEAT[FEAT.date <= cut - pd.Timedelta(days=H + 1)]
        self.test = FEAT[FEAT.date > cut]

    def test_seasonal_naive_uses_same_weekday_last_week(self):
        pred = SeasonalNaive().predict(self.test)
        self.assertTrue(np.array_equal(pred, self.test["lag_7"].to_numpy(float)))

    def test_gradient_boosting_beats_naive_and_is_calibrated_in_level(self):
        m = make_hgb(max_iter=150).fit(self.train, self.train.units)
        p = m.predict(self.test)
        naive = wape(self.test.units, SeasonalNaive().predict(self.test))
        self.assertLess(wape(self.test.units, p), naive * 0.8)
        self.assertLess(abs(bias(self.test.units, p)), 0.05)

    def test_ridge_is_a_real_baseline(self):
        p = make_ridge().fit(self.train, self.train.units).predict(self.test)
        self.assertLess(wape(self.test.units, p), wape(self.test.units, SeasonalNaive().predict(self.test)))


if __name__ == "__main__":
    unittest.main()
