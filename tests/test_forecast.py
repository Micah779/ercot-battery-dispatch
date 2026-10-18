"""no-peeking checks: a forecast for day D must not change when D's (or later) real-time prices change."""

import numpy as np
import pandas as pd

from ercot_bdm.forecast import FEATURES, add_features, forecast_gbm


def _fake_hourly(days=60, start="2024-12-01"):
    rng = np.random.default_rng(0)
    ts = pd.date_range(start, periods=days * 24, freq="h", tz="US/Central")
    df = pd.DataFrame({"interval_start": ts, "hub": "X"})
    df["date"] = df["interval_start"].dt.date
    df["month"] = df["interval_start"].dt.month
    df["hour"] = df["interval_start"].dt.hour
    df["da_price"] = 30 + 10 * np.sin(df["hour"] / 24 * 2 * np.pi) + rng.normal(0, 3, len(df))
    df["rt_price"] = df["da_price"] + rng.normal(0, 8, len(df))
    return df


def test_features_ignore_same_day_rt():
    df = _fake_hourly()
    target = pd.Timestamp("2025-01-15").date()
    bumped = df.copy()
    bumped.loc[bumped["date"] >= target, "rt_price"] += 1000
    a, b = add_features(df), add_features(bumped)
    day = a["date"] == pd.Timestamp(target)
    pd.testing.assert_frame_equal(a.loc[day, FEATURES], b.loc[day, FEATURES])


def test_gbm_ignores_future_months():
    df = _fake_hourly(days=70)
    bumped = df.copy()
    bumped.loc[bumped["interval_start"] >= pd.Timestamp("2025-02-01", tz="US/Central"), "rt_price"] += 1000
    a = add_features(df)
    b = add_features(bumped)
    # february rt is bumped in b; january forecasts must not notice
    fa = forecast_gbm(a, 2025)
    fb = forecast_gbm(b, 2025)
    jan = a["date"].between("2025-01-01", "2025-01-31")
    np.testing.assert_allclose(fa[jan], fb[jan])
