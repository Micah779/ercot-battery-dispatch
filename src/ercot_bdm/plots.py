"""Shared plotting helpers for charts and posts."""

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# chart chrome (light mode, for png posts)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# fixed hub colors so a hub is the same color in every chart
HUB_COLORS = {"HB_WEST": "#eb6834", "HB_HOUSTON": "#2a78d6"}
HUB_LABELS = {"HB_WEST": "West hub", "HB_HOUSTON": "Houston hub"}

# one-hue sequential ramp (blue, light -> dark) for magnitudes
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

MONTHS = [calendar.month_abbr[m] for m in range(1, 13)]


def style():
    """recessive axes, clean defaults."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
        }
    )


MARKET_NAMES = {"rt_price": "real-time", "da_price": "day-ahead"}

# shared footnote lines
NOTE_HOURS = "Hours are hour beginning, Central time: the 0:00 row is ERCOT HE 1 (midnight-1am). DST days kept as-is (23 and 25 hours)."
NOTE_RTCB = "RTC+B went live Dec 5, 2025, so December spans two market designs."


def footnote(fig, lines: list[str]):
    """small muted notes under the chart."""
    fig.text(0.06, -0.01, "\n".join(lines), ha="left", va="top", fontsize=8, color=MUTED)


def hour_grid(df: pd.DataFrame, hub: str, price_col: str, stat: str = "mean") -> pd.DataFrame:
    """hour x month table of a price statistic for one hub."""
    return df[df["hub"] == hub].pivot_table(index="hour", columns="month", values=price_col, aggfunc=stat)


def price_heatmap(
    df: pd.DataFrame,
    hubs: list[str],
    price_col: str = "rt_price",
    year: int | None = None,
    vmin: float = 0,
    vmax: float | None = None,
    stat: str = "mean",
):
    """price by hour of day x month, one panel per hub. pass vmin/vmax to share a scale across charts."""
    style()
    grids = {hub: hour_grid(df, hub, price_col, stat) for hub in hubs}
    if vmax is None:
        vmax = max(g.values.max() for g in grids.values())

    fig, axes = plt.subplots(1, len(hubs), figsize=(5 * len(hubs) + 1, 6.5), sharey=True)
    for ax, hub in zip(np.atleast_1d(axes), hubs):
        im = ax.imshow(grids[hub], aspect="auto", cmap=BLUES, vmin=vmin, vmax=vmax)
        ax.set_title(HUB_LABELS.get(hub, hub))
        ax.set_xticks(range(12), MONTHS, fontsize=8)
        ax.set_yticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Hour beginning (Central)")

    market = MARKET_NAMES[price_col]
    stat_name = "Average" if stat == "mean" else stat.capitalize()
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(f"{stat_name} {market} price ($/MWh)")
    cbar.outline.set_visible(False)
    fig.suptitle(
        f"{stat_name} {market} price by hour and month, ERCOT{f' {year}' if year else ''}",
        x=0.06, ha="left", fontweight="bold", fontsize=13,
    )

    notes = [NOTE_HOURS, NOTE_RTCB]
    if price_col == "rt_price":
        # spikes only dominate in rt; da cell medians sit within ~25% of the mean
        if stat == "mean":
            notes.insert(0, "Averages: a few scarcity hours drive the darkest cells (their medians are half or less).")
        notes.insert(0, "Real-time = hourly average of the four 15-minute settlement prices.")
    footnote(fig, notes)
    return fig


def negative_hours(df: pd.DataFrame, hubs: list[str], price_col: str = "rt_price", year: int | None = None):
    """count of negative-price hours by hour of day and by month, hubs side by side."""
    style()
    neg = df[df[price_col] < 0]
    by_hour = neg.groupby(["hour", "hub"]).size().unstack("hub").reindex(range(24), fill_value=0).fillna(0)
    by_month = neg.groupby(["month", "hub"]).size().unstack("hub").reindex(range(1, 13), fill_value=0).fillna(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1.2]})
    width = 0.4
    for i, hub in enumerate(hubs):
        offset = (i - (len(hubs) - 1) / 2) * width
        total = int(by_hour.get(hub, pd.Series(0)).sum())
        label = f"{HUB_LABELS.get(hub, hub)} ({total} hrs)"
        for ax, data, x in ((ax1, by_hour, np.arange(24)), (ax2, by_month, np.arange(12))):
            vals = data[hub] if hub in data else np.zeros(len(x))
            # small surface gap between neighboring bars
            ax.bar(x + offset, vals, width, color=HUB_COLORS.get(hub), edgecolor=SURFACE, linewidth=1,
                   label=label if ax is ax1 else None)

    ax1.set_title("By hour of day")
    ax1.set_xticks(range(0, 24, 3), [f"{h}:00" for h in range(0, 24, 3)])
    ax1.set_ylabel("Hours with negative price")
    ax1.legend(loc="upper left")
    ax2.set_title("By month")
    ax2.set_xticks(range(12), MONTHS, fontsize=8)
    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    market = MARKET_NAMES[price_col]
    fig.suptitle(
        f"Negative {market} price hours, West vs. Houston{f' ({year})' if year else ''}",
        x=0.06, ha="left", fontweight="bold", fontsize=13,
    )
    fig.tight_layout()

    notes = [NOTE_HOURS, NOTE_RTCB]
    if price_col == "rt_price":
        # say exactly what counts as a negative real-time hour
        any_neg = df[df["rt_min_15"] < 0].groupby("hub").size()
        alt = ", ".join(f"{HUB_LABELS.get(h, h)} {int(any_neg.get(h, 0))}" for h in hubs)
        notes.insert(0, f"Negative hour = hourly average of the four 15-minute prices below $0. Counting any negative interval instead: {alt}.")
    footnote(fig, notes)
    return fig
