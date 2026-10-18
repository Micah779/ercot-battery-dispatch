"""Stage 2: perfect-foresight LP dispatch optimizer."""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from ercot_bdm.config import BATTERY, Battery
from ercot_bdm.settle import revenue


@dataclass
class DayResult:
    charge: np.ndarray  # grid-side charge, mw per hour
    discharge: np.ndarray  # grid-side discharge, mw per hour
    soc: np.ndarray  # stored energy at end of each hour, mwh
    revenue: float  # market $ for the day (price x energy, before cycling cost)
    cycling_cost: float  # $ of cycle cost charged on discharge
    soc_cap_dual: float  # $ per extra mwh of soc ceiling, summed over the day's hours
    cycle_dual: float  # $ per extra mwh of daily discharge allowance
    power_dual: float  # $ per extra mw of charge + discharge rating


class DailyOptimizer:
    """one-day lp: pick hourly charge/discharge to maximize revenue, given known prices.

    efficiency is split evenly: sqrt(rte) on the way in, sqrt(rte) on the way out.
    battery starts and ends each day empty, so days are independent.
    a cycle cost (if set) is subtracted in the objective, so small spreads stop being worth trading.
    """

    def __init__(self, battery: Battery = BATTERY):
        self.b = battery
        self.eta = np.sqrt(battery.round_trip_eff)
        # cache one compiled problem per day length (23, 24, 25 hours on dst days)
        self._problems = {}

    def _build(self, n: int):
        b, eta = self.b, self.eta
        price = cp.Parameter(n)
        c = cp.Variable(n, nonneg=True)  # charge from grid
        d = cp.Variable(n, nonneg=True)  # discharge to grid
        soc = cp.Variable(n, nonneg=True)  # energy stored at end of hour

        # energy balance: soc rises by what goes in after losses, falls by what leaves before losses
        soc_prev = cp.hstack([0, soc[:-1]])  # start empty
        balance = soc == soc_prev + eta * c - d / eta

        soc_cap = soc <= b.energy_mwh  # can't overfill
        charge_cap = c <= b.power_mw  # inverter limits
        discharge_cap = d <= b.power_mw
        # one cycle a day: energy pulled out of the cells <= capacity
        cycle_cap = cp.sum(d) / eta <= b.max_cycles_per_day * b.energy_mwh

        cons = [balance, soc_cap, charge_cap, discharge_cap, cycle_cap, soc[-1] == 0]
        # objective: market revenue minus cycling cost on every mwh discharged
        prob = cp.Problem(cp.Maximize(price @ (d - c) - b.cycle_cost * cp.sum(d)), cons)
        self._problems[n] = (prob, price, c, d, soc, soc_cap, cycle_cap, charge_cap, discharge_cap)

    def solve_day(self, prices) -> DayResult:
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        if n not in self._problems:
            self._build(n)
        prob, price, c, d, soc, soc_cap, cycle_cap, charge_cap, discharge_cap = self._problems[n]

        price.value = prices
        prob.solve(solver=cp.HIGHS)
        if prob.status != cp.OPTIMAL:
            raise RuntimeError(f"solver status {prob.status}")

        # clean tiny solver noise
        cv, dv, sv = (np.clip(x.value, 0, None).round(6) for x in (c, d, soc))
        return DayResult(
            charge=cv,
            discharge=dv,
            soc=sv,
            revenue=float(prices @ (dv - cv)),
            cycling_cost=float(self.b.cycle_cost * dv.sum()),
            soc_cap_dual=float(np.sum(soc_cap.dual_value)),
            cycle_dual=float(cycle_cap.dual_value),
            power_dual=float(np.sum(charge_cap.dual_value) + np.sum(discharge_cap.dual_value)),
        )


def run_year(hourly: pd.DataFrame, hub: str, price_col: str, battery: Battery = BATTERY, settle_col: str | None = None):
    """solve every day for one hub + market. returns (daily summary, hourly dispatch).

    settle_col: plan on price_col (e.g. a forecast) but get paid at settle_col (e.g. rt_price).
    left as none, the plan is settled at the prices it was planned on (perfect foresight).
    """
    settle_col = settle_col or price_col
    opt = DailyOptimizer(battery)
    df = hourly[hourly["hub"] == hub].sort_values("interval_start")

    daily, dispatch = [], []
    for date, day in df.groupby("date", sort=True):
        r = opt.solve_day(day[price_col].to_numpy())
        settle = day[settle_col].to_numpy()
        daily.append(
            {
                "date": date,
                "revenue": revenue(r.charge, r.discharge, settle),
                # what the plan expected to earn on its own prices
                "planned_revenue": r.revenue,
                "cycling_cost": r.cycling_cost,
                "discharged_mwh": r.discharge.sum(),
                "charged_mwh": r.charge.sum(),
                "soc_cap_dual": r.soc_cap_dual,
                "cycle_dual": r.cycle_dual,
                "power_dual": r.power_dual,
                # hours doing both at once (lp can do this to burn energy at negative prices)
                "simultaneous_hours": int(np.sum((r.charge > 1e-3) & (r.discharge > 1e-3))),
            }
        )
        dispatch.append(
            pd.DataFrame(
                {
                    "interval_start": day["interval_start"].to_numpy(),
                    "price": settle,
                    "planned_price": day[price_col].to_numpy(),
                    "charge_mw": r.charge,
                    "discharge_mw": r.discharge,
                    "soc_mwh": r.soc,
                }
            )
        )

    market = settle_col.removesuffix("_price")
    daily = pd.DataFrame(daily).assign(hub=hub, market=market, plan=price_col, cycle_cost=battery.cycle_cost)
    dispatch = pd.concat(dispatch, ignore_index=True).assign(hub=hub, market=market, plan=price_col)

    # value of 1 more mwh of storage: soc ceiling and the daily cycle allowance both scale with it
    daily["capacity_value"] = daily["soc_cap_dual"] + battery.max_cycles_per_day * daily["cycle_dual"]
    return daily, dispatch
