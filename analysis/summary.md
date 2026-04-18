# Round 2 data — summary stats

## Per-day per-product

| product | day | ticks | mid mean | mid std | mid min | mid max | avg spread | trades | vol | vwap |
|---|---|---|---|---|---|---|---|---|---|---|
| ASH_COATED_OSMIUM | -1 | 10000 | 10000.83 | 4.47 | 9981.0 | 10020.0 | 16.22 | 459 | 2348 | 10001.05 |
| ASH_COATED_OSMIUM | 0 | 10000 | 10001.61 | 5.66 | 9979.0 | 10023.0 | 16.25 | 471 | 2404 | 10001.11 |
| ASH_COATED_OSMIUM | 1 | 10000 | 10000.21 | 5.02 | 9980.0 | 10019.0 | 16.23 | 465 | 2375 | 10000.29 |
| INTARIAN_PEPPER_ROOT | -1 | 10000 | 11500.12 | 288.65 | 10998.0 | 12001.5 | 13.07 | 331 | 1669 | 11542.42 |
| INTARIAN_PEPPER_ROOT | 0 | 10000 | 12499.87 | 288.61 | 11996.0 | 13008.0 | 14.12 | 332 | 1671 | 12538.56 |
| INTARIAN_PEPPER_ROOT | 1 | 10000 | 13500.06 | 288.75 | 12995.0 | 14003.0 | 15.18 | 333 | 1693 | 13524.38 |

## Cross-product correlation (concatenated days -1, 0, 1)

- Mid-price level correlation: **-0.0849**
- Tick-to-tick Δmid correlation: **0.0011**

## Files

- Prices: `prices_round_2_day_{-1,0,1}.csv`
- Trades: `trades_round_2_day_{-1,0,1}.csv`
- Plots in `analysis/plots/`
