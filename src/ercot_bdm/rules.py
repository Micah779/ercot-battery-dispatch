"""Stage 3: simple rule-based strategies (e.g. charge 10am-2pm, discharge 6-9pm)."""

from itertools import product

import numpy as np
import pandas as pd

from ercot_bdm.config import BATTERY, Battery
from ercot_bdm.settle import revenue

# the rule under test: charge 10am-2pm, discharge 6-9pm (hour beginning, so 10-13 and 18-20)
CHARGE_HOURS = (10, 11, 12, 13)
DISCHARGE_HOURS = (18, 19, 20)


def time_of_day_schedule(hours, charge_hours=CHARGE_HOURS, discharge_hours=DISCHARGE_HOURS, battery: Battery = BATTERY):
    """fixed daily schedule: fill the battery evenly over the charge window, empty it evenly over the discharge window.

    ignores prices completely. returns grid-side (charge_mw, discharge_mw) per hour.
    """
    hours = np.asarray(hours)
    eta = np.sqrt(battery.round_trip_eff)
    # cells hold at most one cycle a day
    cell_mwh = battery.energy_mwh * battery.max_cycles_per_day

    is_c = np.isin(hours, charge_hours)
    is_d = np.isin(hours, discharge_hours)
    if is_c.sum() == 0 or is_d.sum() == 0:
        return np.zeros(len(hours)), np.zeros(len(hours))

    # grid energy needed to fill the cells, and what comes back out after losses
    # capped by the power rating if the window is too short
    charge_mw = min(cell_mwh / eta / is_c.sum(), battery.power_mw)
    stored = charge_mw * is_c.sum() * eta
    discharge_mw = min(stored * eta / is_d.sum(), battery.power_mw)

    return np.where(is_c, charge_mw, 0.0), np.where(is_d, discharge_mw, 0.0)


def run_rule_year(hourly: pd.DataFrame, hub: str, price_col: str, charge_hours=CHARGE_HOURS,
                  discharge_hours=DISCHARGE_HOURS, battery: Battery = BATTERY):
    """apply the fixed rule every day. same output shape as optimizer.run_year."""
    df = hourly[hourly["hub"] == hub].sort_values("interval_start")

    daily, dispatch = [], []
    for date, day in df.groupby("date", sort=True):
        prices = day[price_col].to_numpy()
        c, d = time_of_day_schedule(day["hour"].to_numpy(), charge_hours, discharge_hours, battery)
        daily.append({"date": date, "revenue": revenue(c, d, prices), "discharged_mwh": d.sum(), "charged_mwh": c.sum()})
        dispatch.append(pd.DataFrame({
            "interval_start": day["interval_start"].to_numpy(),
            "price": prices,
            "charge_mw": c,
            "discharge_mw": d,
        }))

    market = price_col.removesuffix("_price")
    daily = pd.DataFrame(daily).assign(hub=hub, market=market)
    dispatch = pd.concat(dispatch, ignore_index=True).assign(hub=hub, market=market)
    return daily, dispatch


def best_fixed_window(hourly: pd.DataFrame, hub: str, price_col: str, charge_len: int = 4,
                      discharge_len: int = 3, battery: Battery = BATTERY):
    """hindsight search: which single fixed charge/discharge window would have earned the most all year.

    in-sample, so it flatters the rule. it answers: is the problem the timing or the rigidity?
    """
    df = hourly[hourly["hub"] == hub]
    # summed price by hour of day is enough: a fixed schedule's revenue is linear in prices
    # (approximate on the two dst days; exact revenue comes from run_rule_year afterwards)
    by_hour = df.groupby("hour")[price_col].sum()

    best = None
    for cs, ds in product(range(24 - charge_len + 1), range(24 - discharge_len + 1)):
        ch = tuple(range(cs, cs + charge_len))
        dh = tuple(range(ds, ds + discharge_len))
        # discharge has to come after charge within the day
        if ds < cs + charge_len:
            continue
        c, d = time_of_day_schedule(np.arange(24), ch, dh, battery)
        rev = float(by_hour.to_numpy() @ (d - c))
        if best is None or rev > best[0]:
            best = (rev, ch, dh)
    rev, ch, dh = best
    return ch, dh
