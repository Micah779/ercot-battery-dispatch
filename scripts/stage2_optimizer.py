"""stage 2: perfect-foresight revenue for every day of the year, per hub and market."""

import argparse
from dataclasses import replace

import pandas as pd

from ercot_bdm.config import BATTERY, FIGURES_DIR, HUBS, PROCESSED_DIR, RESULTS_DIR, YEAR
from ercot_bdm.data import load_hourly
from ercot_bdm.optimizer import run_year
from ercot_bdm.plots import dispatch_day, monthly_revenue
from ercot_bdm.settle import per_kw_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--cycle-costs", type=float, nargs="+", default=[0, 5, 10], help="$/mwh discharged")
    args = parser.parse_args()

    hourly = load_hourly(args.year)
    daily_all, dispatch_all = [], []
    for hub in HUBS:
        for col in ("rt_price", "da_price"):
            daily, dispatch = run_year(hourly, hub, col)
            daily_all.append(daily)
            dispatch_all.append(dispatch)
            print(f"{hub} {col}: ${daily['revenue'].sum():,.0f}")

    daily = pd.concat(daily_all, ignore_index=True)
    dispatch = pd.concat(dispatch_all, ignore_index=True)
    # hourly dispatch is big-ish, keep it with the data (ignored by git)
    dispatch.to_parquet(PROCESSED_DIR / f"stage2_dispatch_{args.year}.parquet")
    daily.to_csv(RESULTS_DIR / f"stage2_daily_{args.year}.csv", index=False)

    # one row per hub + market: the upper bound numbers
    summary = daily.groupby(["hub", "market"]).agg(
        revenue=("revenue", "sum"),
        discharged_mwh=("discharged_mwh", "sum"),
        cycles=("discharged_mwh", lambda s: s.sum() / (BATTERY.energy_mwh * BATTERY.round_trip_eff**0.5)),
        days_full_cycle=("cycle_dual", lambda s: int((s > 1e-6).sum())),
        capacity_value=("capacity_value", "sum"),
        simultaneous_hours=("simultaneous_hours", "sum"),
    )
    summary["per_kw_month"] = summary["revenue"].apply(per_kw_month)
    # top 10 days' share of the year: how spiky is the money
    top10 = daily.groupby(["hub", "market"])["revenue"].apply(lambda s: s.nlargest(10).sum() / s.sum())
    summary["top10_days_share"] = top10
    summary = summary.round(3)
    summary.to_csv(RESULTS_DIR / f"stage2_summary_{args.year}.csv")
    print(summary.T)

    # sensitivity: charge a cycling cost per mwh discharged, see how much revenue moves
    rows = []
    for cost in args.cycle_costs:
        battery = replace(BATTERY, cycle_cost=cost)
        for hub in HUBS:
            for col in ("rt_price", "da_price"):
                d, _ = run_year(hourly, hub, col, battery)
                rows.append({
                    "hub": hub,
                    "market": col[:2],
                    "cycle_cost": cost,
                    "revenue": d["revenue"].sum(),
                    "net_of_cycle_cost": d["revenue"].sum() - d["cycling_cost"].sum(),
                    "cycles": d["discharged_mwh"].sum() / (battery.energy_mwh * battery.round_trip_eff**0.5),
                    "days_idle": int((d["discharged_mwh"] < 1e-3).sum()),
                })
    sens = pd.DataFrame(rows)
    # compare to the zero-cost run of the same hub + market
    base = sens[sens["cycle_cost"] == 0].set_index(["hub", "market"])["revenue"]
    sens["revenue_vs_free"] = sens["revenue"] / sens.set_index(["hub", "market"]).index.map(base) - 1
    sens = sens.round(3)
    sens.to_csv(RESULTS_DIR / f"stage2_cycle_cost_sensitivity_{args.year}.csv", index=False)
    print(sens.to_string(index=False))

    # charts
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    monthly_revenue(daily, HUBS, "rt", args.year).savefig(
        FIGURES_DIR / f"stage2_monthly_revenue_rt_{args.year}.png", dpi=200, bbox_inches="tight"
    )
    # example: the median-revenue day at west in rt, i.e. a normal day, not a spike day
    west_rt = daily[(daily["hub"] == "HB_WEST") & (daily["market"] == "rt")]
    typical = west_rt.iloc[(west_rt["revenue"] - west_rt["revenue"].median()).abs().argsort().iloc[0]]["date"]
    dispatch_day(dispatch, "HB_WEST", "rt", typical).savefig(
        FIGURES_DIR / f"stage2_dispatch_example_{args.year}.png", dpi=200, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
