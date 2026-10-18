"""Battery parameters and project-wide settings (hubs, dates, paths)."""

from dataclasses import dataclass
from pathlib import Path

# repo root, so paths work no matter where scripts run from
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"

# last full calendar year of data
YEAR = 2025

# hubs we compare: west = wind-heavy, houston = load-heavy
HUBS = ["HB_WEST", "HB_HOUSTON"]

# ercot reports in central time
TZ = "US/Central"

# rtc+b (real-time co-optimization + batteries) market redesign go-live
RTC_B_GO_LIVE = "2025-12-05"


@dataclass(frozen=True)
class Battery:
    power_mw: float = 100.0  # max charge / discharge rate
    energy_mwh: float = 200.0  # usable capacity (2-hour battery)
    round_trip_eff: float = 0.85  # energy out / energy in
    max_cycles_per_day: float = 1.0  # full discharges allowed per day
    cycle_cost: float = 0.0  # $ per mwh discharged, a stand-in for degradation (0 = free cycling)


BATTERY = Battery()
