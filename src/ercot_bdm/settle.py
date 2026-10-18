"""Settle a dispatch schedule against prices and compute revenue metrics."""

import numpy as np

from ercot_bdm.config import BATTERY, Battery


def revenue(charge_mw, discharge_mw, price) -> float:
    """$ earned over hourly intervals: sell discharge, pay for charge (1 mw for 1 hr = 1 mwh)."""
    charge_mw, discharge_mw, price = map(np.asarray, (charge_mw, discharge_mw, price))
    return float(np.sum(price * (discharge_mw - charge_mw)))


def per_kw_month(total_revenue: float, months: float = 12, battery: Battery = BATTERY) -> float:
    """industry yardstick: $ per kw of power capacity per month (what modo reports)."""
    return total_revenue / (battery.power_mw * 1000) / months
