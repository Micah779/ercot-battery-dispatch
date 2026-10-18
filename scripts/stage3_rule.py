"""stage 3: how much of perfect-foresight revenue does a fixed 'charge midday, discharge evening' rule capture?"""

import argparse

import pandas as pd

from ercot_bdm.config import FIGURES_DIR, HUBS, RESULTS_DIR, YEAR
from ercot_bdm.data import load_hourly
from ercot_bdm.optimizer import run_year
from ercot_bdm.plots import _hours_label, capture_by_month, cumulative_revenue
from ercot_bdm.rules import CHARGE_HOURS, DISCHARGE_HOURS, best_fixed_window, run_rule_year
from ercot_bdm.settle import per_kw_month


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=YEAR)
    args = parser.parse_args()

    hourly = load_hourly(args.year)
    opt_all, rule_all, opt_dispatch_all, rows = [], [], [], []
    for hub in HUBS:
        for col in ("rt_price", "da_price"):
            market = col.removesuffix("_price")
            opt, opt_dispatch = run_year(hourly, hub, col)
            rule, _ = run_rule_year(hourly, hub, col)
            # fairer version of the rule: best single fixed window, picked with hindsight
            ch, dh = best_fixed_window(hourly, hub, col)
            best, _ = run_rule_year(hourly, hub, col, ch, dh)
            opt_all.append(opt)
            rule_all.append(rule)
            opt_dispatch_all.append(opt_dispatch)

            total = opt["revenue"].sum()
            rows.append({
                "hub": hub,
                "market": market,
                "perfect_foresight": total,
                "rule": rule["revenue"].sum(),
                "rule_capture": rule["revenue"].sum() / total,
                "rule_per_kw_month": per_kw_month(rule["revenue"].sum()),
                "rule_losing_days": int((rule["revenue"] < 0).sum()),
                "best_window": f"charge {_hours_label(ch)}, discharge {_hours_label(dh)}",
                "best_window_capture": best["revenue"].sum() / total,
                # days where the rule got less than half of what was possible
                "days_under_50pct": int((rule["revenue"].to_numpy() < 0.5 * opt["revenue"].to_numpy()).sum()),
            })

    opt = pd.concat(opt_all, ignore_index=True)
    rule = pd.concat(rule_all, ignore_index=True)
    opt_dispatch = pd.concat(opt_dispatch_all, ignore_index=True)

    summary = pd.DataFrame(rows).round(3)
    summary.to_csv(RESULTS_DIR / f"stage3_summary_{args.year}.csv", index=False)
    print(summary.T.to_string())

    # monthly capture: where the rule breaks
    m = opt.assign(month=pd.to_datetime(opt["date"]).dt.month).groupby(["hub", "market", "month"])["revenue"].sum()
    r = rule.assign(month=pd.to_datetime(rule["date"]).dt.month).groupby(["hub", "market", "month"])["revenue"].sum()
    monthly = pd.DataFrame({"perfect_foresight": m, "rule": r})
    monthly["capture"] = monthly["rule"] / monthly["perfect_foresight"]
    monthly["gap"] = monthly["perfect_foresight"] - monthly["rule"]

    # why: how much of the optimizer's charging / discharging actually falls inside the rule's windows
    od = opt_dispatch.assign(hour=opt_dispatch["interval_start"].dt.hour, month=opt_dispatch["interval_start"].dt.month)
    od["charge_in_window"] = od["charge_mw"].where(od["hour"].isin(CHARGE_HOURS), 0)
    od["discharge_in_window"] = od["discharge_mw"].where(od["hour"].isin(DISCHARGE_HOURS), 0)
    overlap = od.groupby(["hub", "market", "month"])[["charge_mw", "charge_in_window", "discharge_mw", "discharge_in_window"]].sum()
    monthly["opt_charge_in_rule_window"] = overlap["charge_in_window"] / overlap["charge_mw"]
    monthly["opt_discharge_in_rule_window"] = overlap["discharge_in_window"] / overlap["discharge_mw"]
    monthly = monthly.round(3).reset_index()
    monthly.to_csv(RESULTS_DIR / f"stage3_monthly_{args.year}.csv", index=False)

    for hub in HUBS:
        worst = monthly[(monthly["hub"] == hub) & (monthly["market"] == "rt")].nsmallest(3, "capture")
        print(f"\n{hub} rt, worst months:\n{worst.drop(columns=['hub', 'market']).to_string(index=False)}")

    # charts
    rule_label = f"charge {_hours_label(CHARGE_HOURS)}, discharge {_hours_label(DISCHARGE_HOURS)}"
    overall = summary[summary["market"] == "rt"].set_index("hub")["rule_capture"].to_dict()
    capture_by_month(monthly, HUBS, "rt", args.year, rule_label, overall).savefig(
        FIGURES_DIR / f"stage3_capture_by_month_rt_{args.year}.png", dpi=200, bbox_inches="tight"
    )
    cumulative_revenue(opt, rule, HUBS, "rt", args.year).savefig(
        FIGURES_DIR / f"stage3_cumulative_rt_{args.year}.png", dpi=200, bbox_inches="tight"
    )


if __name__ == "__main__":
    main()
