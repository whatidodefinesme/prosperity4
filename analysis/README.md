# Round 2 data — full analysis

Data files (saved at repo root):

- `prices_round_2_day_{-1,0,1}.csv` — level-3 order-book snapshots every 100 ms
- `trades_round_2_day_{-1,0,1}.csv` — executed bot trades

Reproduce:

```bash
pip install pandas numpy matplotlib
python3 analysis/analyze_round2.py
```

Plots land in `analysis/plots/`, auto-regenerated stats go to `analysis/summary.md`.

## Products

Round 2 introduces two new products:

| product | role |
|---|---|
| `ASH_COATED_OSMIUM` | stable, mean-reverting around **10 000** |
| `INTARIAN_PEPPER_ROOT` | trending, drifts **+1 000 / day** around a steady ~288 std |

## Per-day stats

| product | day | ticks | mid mean | mid std | mid min | mid max | avg spread | trades | vol | VWAP |
|---|---|---|---|---|---|---|---|---|---|---|
| ASH_COATED_OSMIUM | -1 | 10 000 | 10 000.83 | 4.47 | 9 981.0 | 10 020.0 | 16.22 | 459 | 2 348 | 10 001.05 |
| ASH_COATED_OSMIUM | 0 | 10 000 | 10 001.61 | 5.66 | 9 979.0 | 10 023.0 | 16.25 | 471 | 2 404 | 10 001.11 |
| ASH_COATED_OSMIUM | 1 | 10 000 | 10 000.21 | 5.02 | 9 980.0 | 10 019.0 | 16.23 | 465 | 2 375 | 10 000.29 |
| INTARIAN_PEPPER_ROOT | -1 | 10 000 | 11 500.12 | 288.65 | 10 998.0 | 12 001.5 | 13.07 | 331 | 1 669 | 11 542.42 |
| INTARIAN_PEPPER_ROOT | 0 | 10 000 | 12 499.87 | 288.61 | 11 996.0 | 13 008.0 | 14.12 | 332 | 1 671 | 12 538.56 |
| INTARIAN_PEPPER_ROOT | 1 | 10 000 | 13 500.06 | 288.75 | 12 995.0 | 14 003.0 | 15.18 | 333 | 1 693 | 13 524.38 |

(Per-product, per-day the number of order-book snapshots is exactly 10 000 — i.e. one every 100 ms for a 1 000 000 ms trading day, consistent with the usual Prosperity 100-ms tick.)

Start / end / drift per day:

| day | product | start | end | drift | σ(Δmid) | AC(Δmid, lag 1) |
|---|---|---:|---:|---:|---:|---:|
| -1 | ASH_COATED_OSMIUM | 9 991 | 10 002 | **+11** | 3.69 | **-0.506** |
| 0 | ASH_COATED_OSMIUM | 10 003 | 10 008 | **+5** | 3.69 | **-0.506** |
| 1 | ASH_COATED_OSMIUM | 10 008 | 9 993 | **-15** | 3.69 | **-0.491** |
| -1 | INTARIAN_PEPPER_ROOT | 11 002 | 12 000 | **+998** | 3.11 | **-0.498** |
| 0 | INTARIAN_PEPPER_ROOT | 11 999 | 13 000 | **+1 001** | 3.32 | **-0.489** |
| 1 | INTARIAN_PEPPER_ROOT | 13 000 | 14 000 | **+999** | 3.58 | **-0.508** |

## Key findings

### 1. `ASH_COATED_OSMIUM` is a textbook "fair-value = 10 000" product
- Mean pinned at 10 000 on every day (±1.6 pt); std ≈ 5.
- Daily range ~±20 around fair value, no trend (drifts on the order of ±10 over an entire 1 M-ms day, i.e. flat noise).
- Average best-bid/ask spread ~**16.2** (wide), top-of-book depth healthy.
- **AC(Δmid, lag 1) ≈ -0.5** → extremely strong short-term mean reversion (classic bid-ask bounce).
- Behaviour is nearly identical to `AMETHYSTS` / `EMERALDS` from prior rounds: pure market-making with a hard-coded fair of 10 000 is the right strategy. Quote both sides inside the book (e.g. 9 998 / 10 002) and let the position limit re-anchor you.

### 2. `INTARIAN_PEPPER_ROOT` has a **deterministic +1 000 / day drift**
- Day -1 starts ~11 000, ends ~12 000. Day 0: 12 000 → 13 000. Day 1: 13 000 → 14 000.
- Within each day the std of the mid is **~288** and the start→end drift is **almost exactly +1 000** — so the price is basically a linear ramp of ~1 pt per 1 000 ms plus noise.
- Tick-to-tick std is only ~3.3, so most of the variance is low-frequency drift rather than high-frequency noise.
- Lag-1 autocorrelation of Δmid is still **~-0.5**, i.e. strong bid-ask bounce at the 100 ms scale even while the longer-horizon trend is positive.
- Spread widens from 13.1 → 14.1 → 15.2 across the three days (mild), top-of-book depth is thinner than Osmium — fewer trades/day (~333 vs ~465) and ~70 % of Osmium's traded volume.

Trading implications:
- **Do NOT fade the trend**. A fixed-fair strategy around the starting level will bleed. Use a slowly-updating fair (e.g. EMA of mid with a long half-life, or `start_of_day + slope · t`).
- Within a single tick, bid-ask-bounce is strong → short-horizon mean-reversion market-making still works *on top of* the drift.
- Expect fair at T+1 000 ms ≈ current_mid + 1 (per-tick drift ≈ 1 / 1 000 × 1 = ~0.001 per ms, but per full day the ramp is +1 000).
- The ramp is so clean that a simple `fair = mid + drift_forecast` works; position must be allowed to run long on up-days.

### 3. Cross-product behaviour
- Level correlation across the concatenated 3 days: **-0.08** (essentially independent).
- Δmid correlation: **~0.00** — the two products are decoupled at the tick level. No useful pairs-trading signal between them.

### 4. Trades file content
- Buyer and seller columns are **entirely blank** (all 2 391 rows). The file only records anonymous bot executions (timestamp, symbol, price, quantity). No counterparty info to exploit.
- `ASH_COATED_OSMIUM` sees ~460 trades and ~2 400 units per day; `INTARIAN_PEPPER_ROOT` sees ~330 trades and ~1 675 units per day.
- VWAP for Pepper Root sits a bit **above** the mean mid on every day (e.g. 11 542 vs 11 500), consistent with the upward drift — actual trades cluster in the later, higher-priced portion of each day.

### 5. Book health notes
- A small number of order-book snapshots have an **empty book** (no bids and no asks, ~100 rows over 60 000 total). The raw CSV reports `mid_price = 0` in those rows. The analysis script masks them out; any trading strategy should also guard against `OrderDepth` being empty.

## Plots

All saved under `analysis/plots/`:

- `midprice_timeseries.png` — stacked mid series for both products across days -1, 0, 1 (vertical dashed lines mark day boundaries). Makes the Osmium mean-revert vs Pepper-Root ramp immediately obvious.
- `spread_depth.png` — top row: best-bid/ask spread over time per product; bottom row: histogram of combined bid1+ask1 volume.
- `return_distribution.png` — log-scale histogram of Δmid per 100 ms tick. Both products are tight and near-symmetric around 0; Osmium has slightly fatter tails.
- `trades_vs_mid.png` — trade scatter (sized by quantity) overlaid on the mid line. Pepper Root's trades visibly climb the ramp.
- `volume_per_day.png` — bar chart of total traded volume per product per day.

## Strategy recommendations for Round 2

1. **`ASH_COATED_OSMIUM`** — reuse the Round 0 / `EMERALDS`-style strategy: fair = 10 000, quote both sides tight (e.g. 9 998 / 10 002, or penny-jump best inside the spread), use position limit to reset. Expected edge: very stable.

2. **`INTARIAN_PEPPER_ROOT`** — dynamic fair. Two viable options:
   - Short-window EMA of mid (e.g. α ≈ 0.05–0.10), then quote fair ± 2–3 ticks. The half-life should be short enough to track the ramp but long enough to filter bid-ask bounce.
   - Parametric ramp: `fair(t) = open_mid + (t / 1_000_000) × 1_000`, updated at the start of each day. Works because the +1 000/day drift is remarkably consistent across all three days observed.
   - Size bias long when position is below limit; the ramp means passive buys held for minutes are in expectation profitable.

3. **Cross-product**: treat independently; no hedging relationship is visible.

4. **Operational**: guard against empty-book ticks (~0.15 % of rows). Do not trade when `OrderDepth.buy_orders` or `OrderDepth.sell_orders` is empty.
