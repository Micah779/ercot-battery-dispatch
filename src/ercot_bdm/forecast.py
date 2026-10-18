"""Stage 4: price forecasts (day-ahead, same hour last week, ML).

timing assumption: the schedule for day D is fixed at midnight starting D.
inputs are prices only: no load, wind or solar data (forecast or actual) is used.
by then we know D's day-ahead prices (published ~1:30pm on D-1) and real-time prices through the end of D-1.
every feature below respects that, so no forecast peeks at D's real-time prices.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ercot_bdm.data import load_hourly

FEATURES = [
    "da_price",  # day-ahead price for this hour, known the day before
    "da_shape",  # this hour's da price minus the day's da mean
    "da_day_max",
    "da_day_min",
    "rt_lag1d",  # real-time, same hour yesterday
    "rt_lag7d",  # real-time, same hour last week
    "rt_prev_mean",  # yesterday's real-time mean / max / min
    "rt_prev_max",
    "rt_prev_min",
    "hour",
    "dow",
    "month",
]


def load_history(years: list[int]) -> pd.DataFrame:
    """several years of hourly prices stacked, so lags and training data reach back before the test year."""
    return pd.concat([load_hourly(y) for y in years], ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """lagged real-time and same-day day-ahead features, one row per hub-hour."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # rt by (hub, date, hour); dst fall-back hour appears twice, average it
    key = df.groupby(["hub", "date", "hour"], as_index=False)["rt_price"].mean()
    for k in (1, 7):
        lag = key.assign(date=key["date"] + pd.Timedelta(days=k)).rename(columns={"rt_price": f"rt_lag{k}d"})
        df = df.merge(lag, on=["hub", "date", "hour"], how="left")

    # yesterday's real-time summary, shifted onto today
    prev = df.groupby(["hub", "date"])["rt_price"].agg(rt_prev_mean="mean", rt_prev_max="max", rt_prev_min="min")
    prev = prev.reset_index().assign(date=lambda x: x["date"] + pd.Timedelta(days=1))
    df = df.merge(prev, on=["hub", "date"], how="left")

    # today's day-ahead shape (all known before the day starts)
    da_day = df.groupby(["hub", "date"])["da_price"]
    df["da_shape"] = df["da_price"] - da_day.transform("mean")
    df["da_day_max"] = da_day.transform("max")
    df["da_day_min"] = da_day.transform("min")

    df["dow"] = df["date"].dt.dayofweek
    return df.sort_values(["hub", "interval_start"]).reset_index(drop=True)


def forecast_last_week(df: pd.DataFrame) -> pd.Series:
    """naive: real-time price at the same hour one week ago (falls back to day-ahead where missing)."""
    return df["rt_lag7d"].fillna(df["da_price"])


def forecast_gbm(df: pd.DataFrame, test_year: int, seed: int = 0) -> pd.Series:
    """gradient boosting, refit each month on everything before that month (walk-forward, no peeking).

    one model per hub. squared loss, so it predicts the expected price, which is what revenue depends on.
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("hub"):
        for month in range(1, 13):
            start = pd.Timestamp(test_year, month, 1)
            end = start + pd.offsets.MonthBegin(1)
            train = g[g["date"] < start]
            test = g[(g["date"] >= start) & (g["date"] < end)]
            if test.empty:
                continue
            # hard guard: training data ends before the test month starts
            assert train["date"].max() < start
            model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
            model.fit(train[FEATURES], train["rt_price"])
            out.loc[test.index] = model.predict(test[FEATURES])
    return out
