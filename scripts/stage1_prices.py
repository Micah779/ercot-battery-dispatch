"""stage 1: pull a year of hub prices, cache them, make the first charts."""

import argparse
import math

from ercot_bdm.config import FIGURES_DIR, HUBS, RESULTS_DIR, YEAR
from ercot_bdm.data import load_hourly
from ercot_bdm.plots import hour_grid, negative_hours, price_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    parser.add_argument("--refresh", action="store_true", help="re-download from ercot")
    args = parser.parse_args()

    df = load_hourly(args.year, refresh=args.refresh)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # one color scale for every heatmap: max cell mean across markets, rounded up to $25
    cell_max = max(hour_grid(df, h, c).values.max() for h in HUBS for c in ("rt_price", "da_price"))
    vmax = math.ceil(cell_max / 25) * 25

    # charts: rt heatmap is the main one, da version for comparison, median as a spike check
    for col, tag in (("rt_price", "rt"), ("da_price", "da")):
        for stat in ("mean", "median"):
            suffix = "" if stat == "mean" else "_median"
            price_heatmap(df, HUBS, col, args.year, vmin=0, vmax=vmax, stat=stat).savefig(
                FIGURES_DIR / f"stage1_heatmap_{tag}{suffix}_{args.year}.png", dpi=200, bbox_inches="tight"
            )
        negative_hours(df, HUBS, col, args.year).savefig(
            FIGURES_DIR / f"stage1_negative_hours_{tag}_{args.year}.png", dpi=200, bbox_inches="tight"
        )

    # summary table for the post
    summary = df.groupby("hub").agg(
        da_mean=("da_price", "mean"),
        rt_mean=("rt_price", "mean"),
        rt_max=("rt_price", "max"),
        rt_min=("rt_price", "min"),
        da_neg_hours=("da_price", lambda s: int((s < 0).sum())),
        # negative hour = hourly avg of 15-min prices below 0
        rt_neg_hours=("rt_price", lambda s: int((s < 0).sum())),
        # stricter alt: any 15-min interval in the hour below 0
        rt_neg_hours_any_interval=("rt_min_15", lambda s: int((s < 0).sum())),
    )
    # night = 10pm-6am, to answer the "printing at night" question directly
    night = df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5])
    summary["rt_neg_hours_night"] = df[night & (df["rt_price"] < 0)].groupby("hub").size()
    summary["rt_neg_hours_10to4"] = df[df["hour"].between(10, 15) & (df["rt_price"] < 0)].groupby("hub").size()
    summary = summary.fillna(0).round(2)
    summary.to_csv(RESULTS_DIR / f"stage1_summary_{args.year}.csv")
    print(summary.T)


if __name__ == "__main__":
    main()
