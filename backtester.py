"""
Local backtester for Prosperity 4 Round 0 trading strategies.
Simulates the exchange matching engine using historical price data.
"""

import csv
import json
import math
from collections import defaultdict
from datamodel import (
    Listing, OrderDepth, Trade, TradingState, Order, Observation
)
from trader import Trader


def load_prices(filename: str):
    """Load price/order book data from CSV."""
    rows = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            rows.append(row)
    return rows


def load_trades(filename: str):
    """Load market trades from CSV."""
    rows = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            rows.append(row)
    return rows


def build_order_depth(price_rows):
    """Build OrderDepth from price data row."""
    od = OrderDepth()
    for i in range(1, 4):
        bp_key = f"bid_price_{i}"
        bv_key = f"bid_volume_{i}"
        ap_key = f"ask_price_{i}"
        av_key = f"ask_volume_{i}"

        if price_rows.get(bp_key) and price_rows[bp_key]:
            bp = int(float(price_rows[bp_key]))
            bv = int(float(price_rows[bv_key]))
            od.buy_orders[bp] = bv

        if price_rows.get(ap_key) and price_rows[ap_key]:
            ap = int(float(price_rows[ap_key]))
            av = int(float(price_rows[av_key]))
            od.sell_orders[ap] = -av  # Sell volumes are negative

    return od


def run_backtest(prices_file: str, trades_file: str, label: str = ""):
    """Run a full backtest on one day of data."""
    price_rows = load_prices(prices_file)
    trade_rows = load_trades(trades_file)

    # Group price data by timestamp
    timestamps_data = defaultdict(dict)
    for row in price_rows:
        ts = int(row['timestamp'])
        product = row['product']
        timestamps_data[ts][product] = row

    # Group trade data by timestamp
    trade_data = defaultdict(lambda: defaultdict(list))
    for row in trade_rows:
        ts = int(float(row['timestamp']))
        sym = row['symbol']
        trade_data[ts][sym].append(row)

    # Get sorted timestamps
    sorted_timestamps = sorted(timestamps_data.keys())

    # Products
    products = set()
    for ts_data in timestamps_data.values():
        products.update(ts_data.keys())
    products = sorted(products)

    # Initialize
    trader = Trader()
    position = {p: 0 for p in products}
    pnl = {p: 0.0 for p in products}
    trader_data = ""
    total_trades = 0
    total_volume = 0

    listings = {}
    for p in products:
        listings[p] = Listing(symbol=p, product=p, denomination="XIRECS")

    prev_own_trades = {p: [] for p in products}

    for i, ts in enumerate(sorted_timestamps):
        ts_data = timestamps_data[ts]

        # Build order depths
        order_depths = {}
        mid_prices = {}
        for p in products:
            if p in ts_data:
                order_depths[p] = build_order_depth(ts_data[p])
                mid_prices[p] = float(ts_data[p]['mid_price'])
            else:
                od = OrderDepth()
                order_depths[p] = od
                mid_prices[p] = 0

        # Build market trades since last iteration
        market_trades = {p: [] for p in products}
        # We use trades from the CSV that happened between previous and current ts
        if i > 0:
            prev_ts = sorted_timestamps[i - 1]
            for t_ts in range(prev_ts + 1, ts + 1):
                if t_ts in trade_data:
                    for sym, tlist in trade_data[t_ts].items():
                        for t in tlist:
                            market_trades[sym].append(
                                Trade(
                                    symbol=sym,
                                    price=int(float(t['price'])),
                                    quantity=int(float(t['quantity'])),
                                    buyer=t.get('buyer', ''),
                                    seller=t.get('seller', ''),
                                    timestamp=int(float(t['timestamp']))
                                )
                            )

        # Build TradingState
        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=prev_own_trades,
            market_trades=market_trades,
            position=dict(position),
            observations=Observation({}, {}),
        )

        # Run trader
        try:
            result, conversions, trader_data = trader.run(state)
        except Exception as e:
            print(f"ERROR at timestamp {ts}: {e}")
            result = {}
            conversions = 0

        # Match orders against the order book
        prev_own_trades = {p: [] for p in products}

        for product, orders in result.items():
            if product not in order_depths:
                continue

            od = order_depths[product]
            limit = trader.POSITION_LIMITS.get(product, 20)

            # Validate: check if all orders combined would exceed position limits
            total_buy_qty = sum(o.quantity for o in orders if o.quantity > 0)
            total_sell_qty = sum(-o.quantity for o in orders if o.quantity < 0)

            if position[product] + total_buy_qty > limit:
                # All buy orders rejected
                orders = [o for o in orders if o.quantity < 0]
                total_buy_qty = 0

            if position[product] - total_sell_qty < -limit:
                # All sell orders rejected
                orders = [o for o in orders if o.quantity > 0]
                total_sell_qty = 0

            for order in orders:
                if order.quantity > 0:
                    # Buy order: match against sell orders
                    sorted_asks = sorted(od.sell_orders.items())
                    for ask_price, ask_vol in sorted_asks:
                        if order.price >= ask_price and order.quantity > 0:
                            # Match
                            available = -ask_vol  # ask_vol is negative
                            fill_qty = min(order.quantity, available)

                            if position[product] + fill_qty > limit:
                                fill_qty = limit - position[product]

                            if fill_qty <= 0:
                                break

                            # Execute trade
                            position[product] += fill_qty
                            pnl[product] -= fill_qty * ask_price
                            order.quantity -= fill_qty
                            od.sell_orders[ask_price] += fill_qty
                            if od.sell_orders[ask_price] == 0:
                                del od.sell_orders[ask_price]

                            total_trades += 1
                            total_volume += fill_qty

                            prev_own_trades[product].append(
                                Trade(product, ask_price, fill_qty,
                                      buyer="SUBMISSION", seller="", timestamp=ts)
                            )

                elif order.quantity < 0:
                    # Sell order: match against buy orders
                    sorted_bids = sorted(od.buy_orders.items(), reverse=True)
                    sell_qty = -order.quantity
                    for bid_price, bid_vol in sorted_bids:
                        if order.price <= bid_price and sell_qty > 0:
                            available = bid_vol
                            fill_qty = min(sell_qty, available)

                            if position[product] - fill_qty < -limit:
                                fill_qty = position[product] + limit

                            if fill_qty <= 0:
                                break

                            # Execute trade
                            position[product] -= fill_qty
                            pnl[product] += fill_qty * bid_price
                            sell_qty -= fill_qty
                            od.buy_orders[bid_price] -= fill_qty
                            if od.buy_orders[bid_price] == 0:
                                del od.buy_orders[bid_price]

                            total_trades += 1
                            total_volume += fill_qty

                            prev_own_trades[product].append(
                                Trade(product, bid_price, fill_qty,
                                      buyer="", seller="SUBMISSION", timestamp=ts)
                            )

    # Final PnL: realized PnL + mark-to-market of remaining position
    print(f"\n{'='*60}")
    print(f"BACKTEST RESULTS: {label}")
    print(f"{'='*60}")

    total_pnl = 0
    for p in products:
        # Get last mid price for mark-to-market
        last_ts = sorted_timestamps[-1]
        if p in timestamps_data[last_ts]:
            last_mid = float(timestamps_data[last_ts][p]['mid_price'])
        else:
            last_mid = 0

        mark_to_market = position[p] * last_mid
        product_pnl = pnl[p] + mark_to_market
        total_pnl += product_pnl

        print(f"\n{p}:")
        print(f"  Final Position: {position[p]}")
        print(f"  Cash PnL (realized): {pnl[p]:.2f}")
        print(f"  Mark-to-Market: {mark_to_market:.2f}")
        print(f"  Total PnL: {product_pnl:.2f}")

    print(f"\n{'='*60}")
    print(f"TOTAL PnL: {total_pnl:.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Volume: {total_volume}")
    print(f"{'='*60}\n")

    return total_pnl


if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))

    pnl_d1 = run_backtest(
        os.path.join(base_dir, "prices_round_0_day_-1.csv"),
        os.path.join(base_dir, "trades_round_0_day_-1.csv"),
        label="Day -1"
    )

    pnl_d2 = run_backtest(
        os.path.join(base_dir, "prices_round_0_day_-2.csv"),
        os.path.join(base_dir, "trades_round_0_day_-2.csv"),
        label="Day -2"
    )

    print(f"\n{'#'*60}")
    print(f"COMBINED PnL (Day -1 + Day -2): {pnl_d1 + pnl_d2:.2f}")
    print(f"Average PnL per day: {(pnl_d1 + pnl_d2) / 2:.2f}")
    print(f"{'#'*60}")
