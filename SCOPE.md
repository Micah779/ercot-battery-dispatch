# ERCOT Battery Dispatch Model — Project Scope

## The question

If you owned a battery in Texas last year, how much money could it have made?
And how close does a simple "charge at noon, sell in the evening" strategy get
to the best possible result?

## The battery we model

- 100 MW power / 200 MWh energy (a 2-hour battery, typical for ERCOT)
- About 85% round-trip efficiency (put in 100 MWh, get back about 85 MWh)
- At most one full cycle per day
- Located at two hubs to compare: **HB_WEST** (lots of wind) and **HB_HOUSTON** (lots of load)

## The stages

Each stage ends with a working result and a post.

### Stage 1: Price data pipeline (week 1)
- Download one year of ERCOT hub prices, both day-ahead and real-time, using `gridstatus`.
- Save it locally (Parquet / DuckDB) so we only download once.
- Charts: a heatmap of average price by hour of day × month, and a count of
  negative-price hours at West vs. Houston.
- **Post:** "Are West Texas wind farms printing at night?" with real data.

### Stage 2: Perfect-foresight optimizer (week 2)
- Write the battery as a linear program (cvxpy or PuLP, HiGHS solver).
- Decisions: how much to charge and discharge each hour.
- Rules: state of charge stays between empty and full, power limits, efficiency losses, one cycle a day.
- Goal: maximize revenue.
- Run it day by day for the whole year. This is the **upper bound**: what you'd earn if you knew prices ahead of time.
- Bonus: the shadow price on the capacity constraint tells you what one more MWh of storage is worth.

### Stage 3: Test the simple rule (week 3)
- Rule: charge 10am–2pm, discharge 6–9pm.
- Compare against the optimizer.
- Result: "The simple rule captures X% of perfect-foresight revenue, and it does worst in these months."
- **Post:** a respectful, data-driven test of Neel's Chapter 10 simplification.

### Stage 4: Realistic dispatch with forecasts (weeks 4–5)
- Real operators don't know prices ahead of time. Plan using a forecast, then get paid real-time prices.
- Forecasts to try, simplest first:
  1. Day-ahead prices
  2. "Same hour last week"
  3. A basic regression or gradient boosting model
- Report the % of perfect-foresight revenue each strategy captures.
- Sanity check: compare $/kW-month against Modo Energy's ERCOT benchmarks.

### Stage 5: Polish
- README: the question, the method, three charts, findings, limitations.
- A write-up linked from my site, and a banner chart for X.

## Out of scope for v1 (listed as "next steps")
- **Ancillary services** (regulation, reserves). Real batteries earn a lot here.
- **Nodal prices**. We use hub prices, not the exact node where a battery would sit.
- **Battery degradation**. We assume cycling doesn't wear the battery down.

Naming what we simplified is part of the point.

## What "done" looks like
One headline number per strategy, in a single table:

| Strategy | Revenue ($) | $/kW-month | % of perfect foresight |
|---|---|---|---|
| Perfect foresight | | | 100% |
| Simple time-of-day rule | | | |
| Day-ahead schedule | | | |
| Same-hour-last-week forecast | | | |
| ML forecast | | | |
