"""sanity checks for the perfect-foresight optimizer."""

import numpy as np
import pytest

from ercot_bdm.config import Battery
from ercot_bdm.optimizer import DailyOptimizer
from ercot_bdm.settle import revenue


def test_flat_prices_earn_nothing():
    # no spread, losses make any trade a loser
    r = DailyOptimizer().solve_day(np.full(24, 30.0))
    assert r.revenue == pytest.approx(0, abs=1e-6)


def test_simple_spread_known_answer():
    # buy 200 mwh at $0, sell 200 * 0.85 = 170 mwh at $100 -> $17,000
    # off-peak at $90 so buying there to sell at $100 loses money after losses
    prices = np.full(24, 90.0)
    prices[2:4] = 0
    prices[18:20] = 100
    r = DailyOptimizer().solve_day(prices)
    assert r.revenue == pytest.approx(17_000, rel=1e-4)


def test_revenue_matches_settlement():
    # objective and the shared settlement function agree
    prices = np.random.default_rng(0).normal(40, 20, 24)
    r = DailyOptimizer().solve_day(prices)
    assert revenue(r.charge, r.discharge, prices) == pytest.approx(r.revenue, rel=1e-6)


def test_limits_hold():
    prices = np.random.default_rng(1).normal(40, 60, 24)
    b = Battery()
    r = DailyOptimizer(b).solve_day(prices)
    assert r.charge.max() <= b.power_mw + 1e-6
    assert r.discharge.max() <= b.power_mw + 1e-6
    assert r.soc.max() <= b.energy_mwh + 1e-6
    assert r.discharge.sum() / np.sqrt(b.round_trip_eff) <= b.energy_mwh + 1e-4


def test_dst_day_lengths():
    # 23- and 25-hour days solve too
    for n in (23, 25):
        assert DailyOptimizer().solve_day(np.linspace(0, 100, n)).revenue > 0


def test_capacity_value_matches_finite_difference():
    # shadow prices should predict what a little more storage is worth
    # (exact only for small steps; lp value is piecewise linear in capacity)
    prices = np.random.default_rng(2).normal(40, 60, 24)
    base = DailyOptimizer(Battery()).solve_day(prices)
    bigger = DailyOptimizer(Battery(energy_mwh=200.1)).solve_day(prices)
    predicted = base.soc_cap_dual + base.cycle_dual
    assert (bigger.revenue - base.revenue) / 0.1 == pytest.approx(predicted, rel=0.01)


def test_cycle_cost_skips_thin_spreads():
    # $30 -> $40 clears 85% losses ($40 * 0.85 = $34 > $30) but not a $10/mwh cycle cost
    prices = np.full(24, 35.0)
    prices[2:4] = 30
    prices[18:20] = 40
    assert DailyOptimizer(Battery()).solve_day(prices).discharge.sum() > 0
    assert DailyOptimizer(Battery(cycle_cost=10)).solve_day(prices).discharge.sum() == pytest.approx(0, abs=1e-6)


def test_cycle_cost_reported_separately():
    # revenue stays market revenue; cost is its own number
    prices = np.random.default_rng(3).normal(40, 60, 24)
    r = DailyOptimizer(Battery(cycle_cost=5)).solve_day(prices)
    assert r.revenue == pytest.approx(revenue(r.charge, r.discharge, prices), rel=1e-6)
    assert r.cycling_cost == pytest.approx(5 * r.discharge.sum(), rel=1e-6)
