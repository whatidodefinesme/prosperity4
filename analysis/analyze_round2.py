"""Round 2 data analysis: loads prices + trades CSVs, plots series, and prints stats.

Outputs PNG plots under analysis/plots/ and a markdown summary at analysis/summary.md.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "analysis" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

DAYS = [-1, 0, 1]
PRODUCTS = ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]
TICKS_PER_DAY = 20_000  # each day = 20k timestamps (0..999900 step 100)


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(REPO / f"prices_round_2_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    # rows with empty books have mid_price==0; treat those as missing
    out.loc[out["bid_price_1"].isna() & out["ask_price_1"].isna(), "mid_price"] = np.nan
    out.loc[out["mid_price"] == 0, "mid_price"] = np.nan
    # continuous timeline across the 3 days (-1, 0, 1)
    out["t"] = out["timestamp"] + (out["day"] - DAYS[0]) * 1_000_000
    return out


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        df = pd.read_csv(REPO / f"trades_round_2_day_{d}.csv", sep=";")
        df["day"] = d
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["t"] = out["timestamp"] + (out["day"] - DAYS[0]) * 1_000_000
    return out


def plot_midprice(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, product in zip(axes, PRODUCTS):
        sub = prices[prices["product"] == product].sort_values("t")
        ax.plot(sub["t"], sub["mid_price"], lw=0.6, color="tab:blue")
        for d in DAYS[1:]:
            ax.axvline((d - DAYS[0]) * 1_000_000, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{product} — mid price (days -1, 0, 1)")
        ax.set_ylabel("mid price")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("timestamp (concatenated across days)")
    fig.tight_layout()
    fig.savefig(OUT / "midprice_timeseries.png", dpi=130)
    plt.close(fig)


def plot_spread_and_depth(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for col, product in enumerate(PRODUCTS):
        sub = prices[prices["product"] == product].copy()
        sub["spread"] = sub["ask_price_1"] - sub["bid_price_1"]
        sub["top_depth"] = sub[["bid_volume_1", "ask_volume_1"]].sum(axis=1)

        ax = axes[0, col]
        ax.plot(sub["t"], sub["spread"], lw=0.5, color="tab:red")
        ax.set_title(f"{product} — best bid/ask spread")
        ax.set_ylabel("spread")
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        ax.hist(sub["top_depth"].dropna(), bins=40, color="tab:green", alpha=0.8)
        ax.set_title(f"{product} — top-of-book depth (bid1+ask1)")
        ax.set_xlabel("combined volume")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "spread_depth.png", dpi=130)
    plt.close(fig)


def plot_returns(prices: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, product in zip(axes, PRODUCTS):
        sub = prices[prices["product"] == product].sort_values("t")
        ret = sub["mid_price"].diff().dropna()
        ax.hist(ret, bins=80, color="tab:purple", alpha=0.85)
        ax.set_title(f"{product} — tick-to-tick Δmid")
        ax.set_xlabel("Δmid (per 100ms tick)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "return_distribution.png", dpi=130)
    plt.close(fig)


def plot_trades(prices: pd.DataFrame, trades: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, product in zip(axes, PRODUCTS):
        p = prices[prices["product"] == product].sort_values("t")
        t = trades[trades["symbol"] == product].sort_values("t")
        ax.plot(p["t"], p["mid_price"], color="lightgrey", lw=0.6, label="mid")
        ax.scatter(t["t"], t["price"], s=np.clip(t["quantity"] * 1.5, 4, 60),
                   c="tab:orange", alpha=0.6, label="trade")
        for d in DAYS[1:]:
            ax.axvline((d - DAYS[0]) * 1_000_000, color="grey", ls="--", lw=0.8)
        ax.set_title(f"{product} — trades vs mid")
        ax.set_ylabel("price")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout()
    fig.savefig(OUT / "trades_vs_mid.png", dpi=130)
    plt.close(fig)


def plot_volume_profile(trades: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, product in zip(axes, PRODUCTS):
        sub = trades[trades["symbol"] == product]
        by_day = sub.groupby("day")["quantity"].sum()
        ax.bar([str(d) for d in by_day.index], by_day.values, color="tab:cyan")
        ax.set_title(f"{product} — total traded volume per day")
        ax.set_ylabel("volume")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "volume_per_day.png", dpi=130)
    plt.close(fig)


def stats_table(prices: pd.DataFrame, trades: pd.DataFrame) -> str:
    lines = ["| product | day | ticks | mid mean | mid std | mid min | mid max | avg spread | trades | vol | vwap |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for product in PRODUCTS:
        for d in DAYS:
            p = prices[(prices["product"] == product) & (prices["day"] == d)]
            t = trades[(trades["symbol"] == product) & (trades["day"] == d)]
            spread = (p["ask_price_1"] - p["bid_price_1"]).mean()
            vwap = (t["price"] * t["quantity"]).sum() / max(t["quantity"].sum(), 1)
            lines.append(
                f"| {product} | {d} | {len(p)} | {p['mid_price'].mean():.2f} | "
                f"{p['mid_price'].std():.2f} | {p['mid_price'].min():.1f} | "
                f"{p['mid_price'].max():.1f} | {spread:.2f} | {len(t)} | "
                f"{int(t['quantity'].sum())} | {vwap:.2f} |"
            )
    return "\n".join(lines)


def correlations(prices: pd.DataFrame) -> tuple[float, float]:
    piv = (prices.pivot_table(index="t", columns="product", values="mid_price")
           .sort_index())
    level_corr = piv.corr().iloc[0, 1]
    ret_corr = piv.diff().corr().iloc[0, 1]
    return level_corr, ret_corr


def main() -> None:
    prices = load_prices()
    trades = load_trades()

    plot_midprice(prices)
    plot_spread_and_depth(prices)
    plot_returns(prices)
    plot_trades(prices, trades)
    plot_volume_profile(trades)

    tbl = stats_table(prices, trades)
    lvl, ret = correlations(prices)

    summary = f"""# Round 2 data — summary stats

## Per-day per-product

{tbl}

## Cross-product correlation (concatenated days -1, 0, 1)

- Mid-price level correlation: **{lvl:.4f}**
- Tick-to-tick Δmid correlation: **{ret:.4f}**

## Files

- Prices: `prices_round_2_day_{{-1,0,1}}.csv`
- Trades: `trades_round_2_day_{{-1,0,1}}.csv`
- Plots in `analysis/plots/`
"""
    (REPO / "analysis" / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
