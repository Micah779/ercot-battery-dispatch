"""sanity checks for the fixed time-of-day rule."""

import numpy as np
import pytest

from ercot_bdm.config import Battery
from ercot_bdm.rules import best_fixed_window, time_of_day_schedule
from ercot_bdm.settle import revenue


def test_rule_fills_and_empties_one_cycle():
    b = Battery()
    c, d = time_of_day_schedule(np.arange(24), battery=b)
    eta = np.sqrt(b.round_trip_eff)
    # 200 mwh into the cells (216.9 from the grid), 184.4 back to the grid: 85% round trip
    assert c.sum() * eta == pytest.approx(200)
    assert d.sum() / c.sum() == pytest.approx(b.round_trip_eff)
    assert c.max() <= b.power_mw and d.max() <= b.power_mw
    # only in the rule's windows
    assert set(np.flatnonzero(c)) == {10, 11, 12, 13}
    assert set(np.flatnonzero(d)) == {18, 19, 20}


def test_rule_loses_on_flat_prices():
    # no spread: rule still trades and eats the losses
    c, d = time_of_day_schedule(np.arange(24))
    assert revenue(c, d, np.full(24, 30.0)) < 0


def test_short_window_capped_by_power():
    # 1-hour charge window can only take 100 mwh
    b = Battery()
    c, d = time_of_day_schedule(np.arange(24), (10,), (18, 19, 20), b)
    assert c.max() == pytest.approx(100)
    assert d.sum() == pytest.approx(100 * b.round_trip_eff)


def test_best_window_finds_obvious_answer():
    import pandas as pd
    # every day cheap at 2-5am, expensive at 7-9pm
    prices = np.full(24, 40.0)
    prices[2:6] = 5
    prices[19:22] = 150
    hourly = pd.DataFrame({"hub": "X", "hour": np.tile(np.arange(24), 3), "p": np.tile(prices, 3)})
    assert best_fixed_window(hourly, "X", "p") == ((2, 3, 4, 5), (19, 20, 21))
