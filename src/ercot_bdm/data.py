"""Stage 1: download ERCOT hub prices with gridstatus and cache them locally."""

import pandas as pd
from gridstatus import Ercot

from ercot_bdm.config import HUBS, PROCESSED_DIR, RAW_DIR, TZ, YEAR

# gridstatus market name -> short name used in file names / columns
MARKETS = {"DAY_AHEAD_HOURLY": "da", "REAL_TIME_15_MIN": "rt"}


def _raw_path(market: str, year: int):
    return RAW_DIR / f"spp_{MARKETS[market]}_{year}.parquet"


def fetch_year(market: str, year: int = YEAR, refresh: bool = False) -> pd.DataFrame:
    """yearly settlement point prices for all hubs + load zones, cached as parquet."""
    path = _raw_path(market, year)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    # yearly historical files from ercot (np4-180 for da, np6-785 for rt)
    ercot = Ercot()
    if market == "DAY_AHEAD_HOURLY":
        df = ercot.get_dam_spp(year)
    else:
        df = ercot.get_rtm_spp(year)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def build_hourly(year: int = YEAR, hubs: list[str] = HUBS, refresh: bool = False) -> pd.DataFrame:
    """one row per hub-hour with da and rt prices side by side."""
    cols = ["Interval Start", "Location", "SPP"]

    da = fetch_year("DAY_AHEAD_HOURLY", year, refresh)[cols]
    da = da[da["Location"].isin(hubs)].rename(columns={"SPP": "da_price"})

    rt = fetch_year("REAL_TIME_15_MIN", year, refresh)[cols]
    rt = rt[rt["Location"].isin(hubs)]
    # rt is 15-min, average the four intervals to the hour
    # floor in utc so the repeated dst hour doesn't get merged
    rt["Interval Start"] = (
        rt["Interval Start"].dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(TZ)
    )
    # keep the hourly min too, to count hours with any negative interval
    rt = rt.groupby(["Interval Start", "Location"], as_index=False)["SPP"].agg(
        rt_price="mean", rt_min_15="min"
    )

    df = da.merge(rt, on=["Interval Start", "Location"], how="outer")
    df = df.rename(columns={"Interval Start": "interval_start", "Location": "hub"})
    df["hub"] = df["hub"].astype(str)

    # local calendar fields for grouping
    # hour is hour beginning: hour 0 = ercot "HE 1" = midnight to 1am
    # dst: march day has 23 hours (no hour 2), november day has 25 (hour 1 twice, both kept)
    df["date"] = df["interval_start"].dt.date
    df["month"] = df["interval_start"].dt.month
    df["hour"] = df["interval_start"].dt.hour

    return df.sort_values(["hub", "interval_start"]).reset_index(drop=True)


def load_hourly(year: int = YEAR, refresh: bool = False) -> pd.DataFrame:
    """cached hourly table; builds it on first call."""
    path = PROCESSED_DIR / f"prices_hourly_{year}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = build_hourly(year, refresh=refresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
