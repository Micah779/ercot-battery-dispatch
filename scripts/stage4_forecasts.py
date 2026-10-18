"""stage 4: schedule with only what's known the night before, get paid real-time prices."""

import argparse

import numpy as np
import pandas as pd

from ercot_bdm.config import FIGURES_DIR, HUBS, MODO_2025_PER_KW_MONTH, RESULTS_DIR, YEAR
from ercot_bdm.forecast import add_features, forecast_gbm, forecast_last_week, load_history
from ercot_bdm.optimizer import run_year
from ercot_bdm.plots import strategy_comparison
from ercot_bdm.rules import run_rule_year
from ercot_bdm.settle import per_kw_month

# peak = 4pm-9pm (hour beginning 16-20), where the evening spread is earned
PEAK_HOURS = range(16, 21)
FORECASTS = ["da_forecast", "gbm_forecast", "last_week_forecast"]
N_BOOT = 5000

# strategy -> column the optimizer plans on (rule has no forecast)
STRATEGIES = {
    "perfect_foresight": "rt_price",
    "da_forecast": "fc_da",
    "gbm_forecast": "fc_gbm",
    "last_week_forecast": "fc_last_week",
    "simple_rule": None,
}
LABELS = {
    "perfect_foresight": "Perfect foresight (upper bound)",
    "da_forecast": "Plan on day-ahead prices",
    "gbm_forecast": "Plan on ML forecast (gradient boosting)",
    "last_week_forecast": "Plan on same hour last week",
    "simple_rule": "Fixed rule (charge 10-14, discharge 18-21)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    args = parser.parse_args()

    # prior year gives lags for early january and training data for the ml model
    df = add_features(load_history([args.year - 1, args.year]))
    df["fc_da"] = df["da_price"]
    df["fc_last_week"] = forecast_last_week(df)
    df["fc_gbm"] = forecast_gbm(df, args.year)
    df = df[df["date"].dt.year == args.year].copy()

    rows, daily_all = [], []
    for hub in HUBS:
        h = df[df["hub"] == hub]
        for name, plan_col in STRATEGIES.items():
            if plan_col is None:
                daily, _ = run_rule_year(df, hub, "rt_price")
            else:
                daily, _ = run_year(df, hub, plan_col, settle_col="rt_price")
            daily_all.append(daily.assign(strategy=name))
            row = {
                "hub": hub,
                "strategy": name,
                "revenue": daily["revenue"].sum(),
                "losing_days": int((daily["revenue"] < 0).sum()),
            }
            if plan_col is not None and plan_col != "rt_price":
                err = (h[plan_col] - h["rt_price"]).abs()
                row["mae"] = err.mean()
                row["mae_peak"] = err[h["hour"].isin(PEAK_HOURS)].mean()
                # median error, less dominated by spikes
                row["median_abs_err"] = err.median()
                # how much the plan thought it would make vs what it made
                row["planned_revenue"] = daily["planned_revenue"].sum()
            rows.append(row)

    summary = pd.DataFrame(rows)
    pf = summary[summary["strategy"] == "perfect_foresight"].set_index("hub")["revenue"]
    summary["capture"] = summary["revenue"] / summary["hub"].map(pf)
    summary["per_kw_month"] = summary["revenue"].apply(per_kw_month)
    summary = summary.round(3)
    summary.to_csv(RESULTS_DIR / f"stage4_summary_{args.year}.csv", index=False)
    print(summary.to_string(index=False))

    # monthly capture per strategy
    daily = pd.concat(daily_all, ignore_index=True)
    daily["month"] = pd.to_datetime(daily["date"]).dt.month
    monthly = daily.groupby(["hub", "strategy", "month"])["revenue"].sum().unstack("strategy")
    monthly = monthly.div(monthly["perfect_foresight"], axis=0).round(3)
    monthly.to_csv(RESULTS_DIR / f"stage4_monthly_capture_{args.year}.csv")
    print("\nmonthly capture (rt):\n", monthly.to_string())

    # error vs money: does the most accurate forecast earn the most?
    err_table = summary[summary["strategy"].isin(FORECASTS)][["hub", "strategy", "mae", "mae_peak", "capture"]]
    err_table.to_csv(RESULTS_DIR / f"stage4_error_vs_capture_{args.year}.csv", index=False)
    print("\nforecast error vs capture:\n", err_table.to_string(index=False))

    # is ml vs day-ahead a real gap? bootstrap the capture difference over days, plus count months won
    wide = daily.pivot_table(index=["hub", "date"], columns="strategy", values="revenue")
    rng = np.random.default_rng(0)
    sig = []
    for hub in HUBS:
        w = wide.loc[hub]
        diff = (w["gbm_forecast"] - w["da_forecast"]).to_numpy()
        pf_total = w["perfect_foresight"].sum()
        # plain day resampling
        idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
        boot = diff[idx].sum(axis=1) / pf_total
        # weekly blocks, since neighboring days are correlated
        weeks = pd.to_datetime(w.index).isocalendar().week.to_numpy()
        week_sums = pd.Series(diff).groupby(weeks).sum().to_numpy()
        widx = rng.integers(0, len(week_sums), (N_BOOT, len(week_sums)))
        wboot = week_sums[widx].sum(axis=1) / pf_total
        m = monthly.loc[hub]
        sig.append({
            "hub": hub,
            "gbm_minus_da_capture": diff.sum() / pf_total,
            "ci95_low_days": np.percentile(boot, 2.5),
            "ci95_high_days": np.percentile(boot, 97.5),
            "ci95_low_weeks": np.percentile(wboot, 2.5),
            "ci95_high_weeks": np.percentile(wboot, 97.5),
            "share_boot_gbm_ahead": (wboot > 0).mean(),
            "months_gbm_wins": int((m["gbm_forecast"] > m["da_forecast"]).sum()),
            "days_gbm_wins": int((diff > 0).sum()),
            "days_tied": int((np.abs(diff) < 1).sum()),
        })
    sig = pd.DataFrame(sig).round(3)
    sig.to_csv(RESULTS_DIR / f"stage4_gbm_vs_da_significance_{args.year}.csv", index=False)
    print("\nml vs day-ahead plan:\n", sig.T.to_string())

    strategy_comparison(summary, HUBS, list(STRATEGIES), LABELS, MODO_2025_PER_KW_MONTH, args.year).savefig(
        FIGURES_DIR / f"stage4_strategy_comparison_{args.year}.png", dpi=200, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
