from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json
import math


class Trader:
    """
    Optimized PnL trading strategy for Prosperity 4 - Round 0.

    Products:
    - EMERALDS: Stable fair value at 10000. Market-make with tight spread.
    - TOMATOES: Mean-reverting. Track fair value with EMA, trade deviations.
    """

    # Position limits per product
    POSITION_LIMITS = {
        "EMERALDS": 50,
        "TOMATOES": 50,
    }

    # EMERALDS: fixed fair value, pure market-making
    EMERALDS_FAIR_VALUE = 10000

    # TOMATOES: EMA-based fair value tracking
    TOMATOES_EMA_SPAN = 58  # EMA span for fair value estimation (optimized via backtesting)
    TOMATOES_EMA_ALPHA = 2.0 / (58 + 1)  # ~0.034

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        # Deserialize trader data (persisted state across iterations)
        trader_state = {}
        if state.traderData and state.traderData != "":
            try:
                trader_state = json.loads(state.traderData)
            except (json.JSONDecodeError, TypeError):
                trader_state = {}

        for product in state.order_depths:
            if product == "EMERALDS":
                orders = self.trade_emeralds(state, product)
            elif product == "TOMATOES":
                orders, trader_state = self.trade_tomatoes(
                    state, product, trader_state
                )
            else:
                orders = []
            result[product] = orders

        trader_data = json.dumps(trader_state)
        conversions = 0
        return result, conversions, trader_data

    def trade_emeralds(self, state: TradingState, product: str) -> List[Order]:
        """
        EMERALDS strategy: Pure market-making around fixed fair value of 10000.

        Key insight: EMERALDS mid price is 10000 in 97% of timestamps.
        Bids are almost always at 9992, asks at 10008 (spread=16).
        We place orders inside that spread to capture profit.

        Approach:
        1. Aggressively take any mispriced orders (buy below fair, sell above fair)
        2. Place limit orders at fair-2 (buy) and fair+2 (sell) to capture spread
        3. Adjust aggressiveness based on position to manage inventory risk
        """
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        fair = self.EMERALDS_FAIR_VALUE

        buy_capacity = limit - position   # max we can buy
        sell_capacity = limit + position  # max we can sell (absolute)

        # --- STEP 1: Take mispriced orders (aggressive) ---

        # Buy all sell orders priced below fair value
        remaining_buy = buy_capacity
        if remaining_buy > 0:
            sorted_asks = sorted(order_depth.sell_orders.items())
            for ask_price, ask_vol in sorted_asks:
                if ask_price < fair and remaining_buy > 0:
                    # ask_vol is negative in sell_orders
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty
                elif ask_price == fair and remaining_buy > 0:
                    # Buy at fair value too, it's free edge
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty

        # Sell into all buy orders priced above fair value
        remaining_sell = sell_capacity
        if remaining_sell > 0:
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for bid_price, bid_vol in sorted_bids:
                if bid_price > fair and remaining_sell > 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty
                elif bid_price == fair and remaining_sell > 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty

        # --- STEP 2: Place limit orders for market making ---
        # Adjust spread based on position to manage inventory
        # If long, we want to sell more aggressively (tighter ask, wider bid)
        # If short, we want to buy more aggressively (tighter bid, wider ask)

        position_ratio = position / limit if limit > 0 else 0

        # Base spread offsets
        buy_offset = 2   # buy at fair - offset
        sell_offset = 2  # sell at fair + offset

        # Adjust offsets based on position (inventory management)
        # When heavily long, buy less aggressively and sell more aggressively
        if position_ratio > 0.5:
            buy_offset = 4
            sell_offset = 1
        elif position_ratio > 0.25:
            buy_offset = 3
            sell_offset = 1
        elif position_ratio < -0.5:
            buy_offset = 1
            sell_offset = 4
        elif position_ratio < -0.25:
            buy_offset = 1
            sell_offset = 3

        # Place buy limit order
        if remaining_buy > 0:
            buy_price = fair - buy_offset
            orders.append(Order(product, buy_price, remaining_buy))

        # Place sell limit order
        if remaining_sell > 0:
            sell_price = fair + sell_offset
            orders.append(Order(product, sell_price, -remaining_sell))

        return orders

    def trade_tomatoes(
        self, state: TradingState, product: str, trader_state: dict
    ) -> tuple:
        """
        TOMATOES strategy: EMA-based mean-reversion market making.

        Key insight: TOMATOES has strong mean-reversion (autocorr = -0.41).
        Price oscillates around a slowly moving fair value.

        Approach:
        1. Track fair value using EMA of mid prices
        2. Take all orders on the wrong side of fair value
        3. Place limit orders around fair value with position-aware spread
        4. More aggressive when position is flat, more defensive when skewed
        """
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        # Calculate current mid price from order book
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        if best_bid is None or best_ask is None:
            return orders, trader_state

        mid_price = (best_bid + best_ask) / 2.0

        # Update EMA fair value
        ema_key = "tomatoes_ema"
        if ema_key in trader_state:
            prev_ema = trader_state[ema_key]
            alpha = self.TOMATOES_EMA_ALPHA
            fair_value = alpha * mid_price + (1 - alpha) * prev_ema
        else:
            fair_value = mid_price

        trader_state[ema_key] = fair_value

        # Round fair value for integer price orders
        fair_int = round(fair_value)

        buy_capacity = limit - position
        sell_capacity = limit + position

        # --- STEP 1: Aggressively take mispriced orders ---

        # Buy all sell orders below fair value
        remaining_buy = buy_capacity
        if remaining_buy > 0:
            sorted_asks = sorted(order_depth.sell_orders.items())
            for ask_price, ask_vol in sorted_asks:
                if ask_price < fair_value and remaining_buy > 0:
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty
                elif ask_price == fair_int and remaining_buy > 0:
                    # At fair value, buy if we're short or flat
                    if position <= 0:
                        take_qty = min(-ask_vol, remaining_buy)
                        orders.append(Order(product, ask_price, take_qty))
                        remaining_buy -= take_qty

        # Sell into all buy orders above fair value
        remaining_sell = sell_capacity
        if remaining_sell > 0:
            sorted_bids = sorted(order_depth.buy_orders.items(), reverse=True)
            for bid_price, bid_vol in sorted_bids:
                if bid_price > fair_value and remaining_sell > 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty
                elif bid_price == fair_int and remaining_sell > 0:
                    # At fair value, sell if we're long or flat
                    if position >= 0:
                        take_qty = min(bid_vol, remaining_sell)
                        orders.append(Order(product, bid_price, -take_qty))
                        remaining_sell -= take_qty

        # --- STEP 2: Market-making limit orders ---
        position_ratio = position / limit if limit > 0 else 0

        # Dynamic spread based on position
        # Tighter spread when flat, wider when skewed
        if abs(position_ratio) > 0.6:
            buy_offset = 4 if position_ratio > 0 else 1
            sell_offset = 1 if position_ratio > 0 else 4
        elif abs(position_ratio) > 0.3:
            buy_offset = 3 if position_ratio > 0 else 1
            sell_offset = 1 if position_ratio > 0 else 3
        else:
            buy_offset = 2
            sell_offset = 2

        # Volume scaling: post more when flat, less when skewed
        buy_vol = remaining_buy
        sell_vol = remaining_sell

        # Place buy limit order
        if buy_vol > 0:
            buy_price = fair_int - buy_offset
            orders.append(Order(product, buy_price, buy_vol))

        # Place sell limit order
        if sell_vol > 0:
            sell_price = fair_int + sell_offset
            orders.append(Order(product, sell_price, -sell_vol))

        return orders, trader_state
