from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class Trader:
    """
    Optimized PnL trading strategy for Prosperity 4 - Round 0.

    Products:
    - EMERALDS: Stable fair value at 10000. Penny-jumping market-making.
    - TOMATOES: Mean-reverting. EMA(58) fair value + tight market-making.

    Key insight: most PnL comes from passive fills (bots trading against
    our limit orders). Tighter quotes = higher fill probability = more PnL.
    """

    POSITION_LIMITS = {
        "EMERALDS": 50,
        "TOMATOES": 50,
    }

    EMERALDS_FAIR_VALUE = 10000

    TOMATOES_EMA_SPAN = 58
    TOMATOES_EMA_ALPHA = 2.0 / (58 + 1)

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
        EMERALDS: Penny-jumping market-making around fair value 10000.

        Strategy:
        1. Aggressively take any mispriced orders at or through fair value
        2. Post penny-jump quotes (best_bid+1 / best_ask-1) to be the
           tightest quote in the book, maximizing bot fill probability
        3. Skew quotes based on position to flatten inventory
        """
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        fair = self.EMERALDS_FAIR_VALUE

        buy_capacity = limit - position
        sell_capacity = limit + position

        # --- STEP 1: Sweep all mispriced orders ---
        remaining_buy = buy_capacity
        if remaining_buy > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price < fair and remaining_buy > 0:
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty
                elif ask_price == fair and remaining_buy > 0:
                    # At fair: buy if short or flat to flatten position
                    if position <= 0:
                        take_qty = min(-ask_vol, remaining_buy)
                        orders.append(Order(product, ask_price, take_qty))
                        remaining_buy -= take_qty

        remaining_sell = sell_capacity
        if remaining_sell > 0:
            for bid_price, bid_vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if bid_price > fair and remaining_sell > 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty
                elif bid_price == fair and remaining_sell > 0:
                    if position >= 0:
                        take_qty = min(bid_vol, remaining_sell)
                        orders.append(Order(product, bid_price, -take_qty))
                        remaining_sell -= take_qty

        # --- STEP 2: Penny-jump market-making with position skew ---
        best_bid = (
            max(order_depth.buy_orders.keys())
            if order_depth.buy_orders
            else fair - 8
        )
        best_ask = (
            min(order_depth.sell_orders.keys())
            if order_depth.sell_orders
            else fair + 8
        )

        # Position-based offset: tighter when flat, wider when skewed
        # position_ratio: -1 (max short) to +1 (max long)
        pos_ratio = position / limit if limit > 0 else 0.0

        # Base: penny-jump (1 tick inside best bid/ask)
        # Skew: when long, buy less aggressively and sell more aggressively
        buy_price = best_bid + 1
        sell_price = best_ask - 1

        # Position skewing: shift prices to flatten inventory
        if pos_ratio > 0.4:
            # Heavily long: widen buy, tighten sell toward fair
            buy_price = min(buy_price, fair - 2)
            sell_price = min(sell_price, fair + 1)
        elif pos_ratio > 0.2:
            buy_price = min(buy_price, fair - 1)
            sell_price = min(sell_price, fair + 1)
        elif pos_ratio < -0.4:
            # Heavily short: tighten buy toward fair, widen sell
            buy_price = max(buy_price, fair - 1)
            sell_price = max(sell_price, fair + 2)
        elif pos_ratio < -0.2:
            sell_price = max(sell_price, fair + 1)
            buy_price = max(buy_price, fair - 1)

        # Clamp: never buy above fair or sell below fair
        buy_price = min(buy_price, fair - 1)
        sell_price = max(sell_price, fair + 1)

        if remaining_buy > 0:
            orders.append(Order(product, buy_price, remaining_buy))

        if remaining_sell > 0:
            orders.append(Order(product, sell_price, -remaining_sell))

        return orders

    def trade_tomatoes(
        self, state: TradingState, product: str, trader_state: dict
    ) -> tuple:
        """
        TOMATOES: EMA(58)-based mean-reversion with tight market-making.

        Strong negative autocorrelation (-0.41) means prices revert to EMA.
        We aggressively take mispriced orders and post tight penny-jump
        quotes around the EMA fair value to maximize passive fill rate.
        """
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]

        best_bid = (
            max(order_depth.buy_orders.keys())
            if order_depth.buy_orders
            else None
        )
        best_ask = (
            min(order_depth.sell_orders.keys())
            if order_depth.sell_orders
            else None
        )

        if best_bid is None or best_ask is None:
            return orders, trader_state

        mid_price = (best_bid + best_ask) / 2.0

        # Update EMA fair value
        ema_key = "tomatoes_ema"
        if ema_key in trader_state:
            prev_ema = trader_state[ema_key]
            fair_value = (
                self.TOMATOES_EMA_ALPHA * mid_price
                + (1 - self.TOMATOES_EMA_ALPHA) * prev_ema
            )
        else:
            fair_value = mid_price

        trader_state[ema_key] = fair_value
        fair_int = round(fair_value)

        buy_capacity = limit - position
        sell_capacity = limit + position

        # --- STEP 1: Aggressively sweep mispriced orders ---
        remaining_buy = buy_capacity
        if remaining_buy > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if remaining_buy <= 0:
                    break
                if ask_price < fair_value:
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty
                elif ask_price <= fair_int and position <= 0:
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty

        remaining_sell = sell_capacity
        if remaining_sell > 0:
            for bid_price, bid_vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if remaining_sell <= 0:
                    break
                if bid_price > fair_value:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty
                elif bid_price >= fair_int and position >= 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty

        # --- STEP 2: Tight market-making with position skew ---
        pos_ratio = position / limit if limit > 0 else 0.0

        # Penny-jump: post 1 tick inside current best bid/ask
        buy_price = best_bid + 1
        sell_price = best_ask - 1

        # Position-aware skewing
        if pos_ratio > 0.4:
            # Heavily long: widen buy, tighten sell
            buy_price = min(buy_price, fair_int - 2)
            sell_price = min(sell_price, fair_int + 1)
        elif pos_ratio > 0.2:
            buy_price = min(buy_price, fair_int - 1)
            sell_price = min(sell_price, fair_int + 1)
        elif pos_ratio < -0.4:
            # Heavily short: tighten buy, widen sell
            buy_price = max(buy_price, fair_int - 1)
            sell_price = max(sell_price, fair_int + 2)
        elif pos_ratio < -0.2:
            buy_price = max(buy_price, fair_int - 1)
            sell_price = max(sell_price, fair_int + 1)

        # Clamp: never buy above fair or sell below fair
        buy_price = min(buy_price, fair_int - 1)
        sell_price = max(sell_price, fair_int + 1)

        if remaining_buy > 0:
            orders.append(Order(product, buy_price, remaining_buy))

        if remaining_sell > 0:
            orders.append(Order(product, sell_price, -remaining_sell))

        return orders, trader_state
