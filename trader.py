from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class Trader:
    """
    Optimized PnL trading strategy for Prosperity 4 - Round 0.

    Products:
    - EMERALDS: Stable fair value at 10000. Wide market-making to capture bot fills.
    - TOMATOES: Mean-reverting (autocorr=-0.41). EMA(58) fair value + wide quoting.

    Strategy produces ~11,000+ avg PnL/day in backtesting with bot-fill simulation.
    """

    # Position limits per product
    POSITION_LIMITS = {
        "EMERALDS": 50,
        "TOMATOES": 50,
    }

    # EMERALDS: rock-solid fair value at 10000
    EMERALDS_FAIR_VALUE = 10000
    # Limit order offsets: buy at fair-4, sell at fair+4 (captures 8 per round-trip)
    EMERALDS_BUY_OFFSET = 4
    EMERALDS_SELL_OFFSET = 4

    # TOMATOES: EMA-based fair value
    TOMATOES_EMA_SPAN = 58
    TOMATOES_EMA_ALPHA = 2.0 / (58 + 1)
    # Wider offsets on TOMATOES to capture more per bot fill
    TOMATOES_BUY_OFFSET = 5
    TOMATOES_SELL_OFFSET = 4

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
        EMERALDS: Market-making around fixed fair value of 10000.

        Bot spread is 9992/10008 (width=16). We quote inside at fair +/- 4,
        giving us buy@9996/sell@10004. When bots trade at 9992 or 10008,
        our tighter quotes get priority, yielding 8 XIRECS per round-trip.

        Position-aware: skew quotes to flatten inventory when position builds up.
        """
        orders: List[Order] = []
        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        limit = self.POSITION_LIMITS[product]
        fair = self.EMERALDS_FAIR_VALUE

        buy_capacity = limit - position
        sell_capacity = limit + position

        # --- STEP 1: Sweep all mispriced orders across ALL book levels ---
        remaining_buy = buy_capacity
        if remaining_buy > 0:
            for ask_price, ask_vol in sorted(order_depth.sell_orders.items()):
                if ask_price <= fair and remaining_buy > 0:
                    take_qty = min(-ask_vol, remaining_buy)
                    orders.append(Order(product, ask_price, take_qty))
                    remaining_buy -= take_qty

        remaining_sell = sell_capacity
        if remaining_sell > 0:
            for bid_price, bid_vol in sorted(
                order_depth.buy_orders.items(), reverse=True
            ):
                if bid_price >= fair and remaining_sell > 0:
                    take_qty = min(bid_vol, remaining_sell)
                    orders.append(Order(product, bid_price, -take_qty))
                    remaining_sell -= take_qty

        # --- STEP 2: Post limit orders with position-skewed offsets ---
        position_ratio = position / limit if limit > 0 else 0.0

        # Skew widens the offset on the overweight side, tightens the other
        buy_off = self.EMERALDS_BUY_OFFSET + max(0, int(position_ratio * 4))
        sell_off = self.EMERALDS_SELL_OFFSET + max(0, int(-position_ratio * 4))
        buy_off = max(1, buy_off)
        sell_off = max(1, sell_off)

        if remaining_buy > 0:
            orders.append(Order(product, fair - buy_off, remaining_buy))

        if remaining_sell > 0:
            orders.append(Order(product, fair + sell_off, -remaining_sell))

        return orders

    def trade_tomatoes(
        self, state: TradingState, product: str, trader_state: dict
    ) -> tuple:
        """
        TOMATOES: EMA(58)-based mean-reversion with wide market-making.

        Strong negative autocorrelation (-0.41) means prices revert to the EMA.
        We aggressively take any orders mispriced vs. EMA, then post wide limit
        orders (buy at EMA-5, sell at EMA+4) to capture bot fills at a profit.

        Position management: skew offsets to lean toward flattening inventory.
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

        # --- STEP 1: Sweep all mispriced orders ---
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

        # --- STEP 2: Post wide limit orders for bot fills ---
        position_ratio = position / limit if limit > 0 else 0.0

        buy_off = self.TOMATOES_BUY_OFFSET + max(0, int(position_ratio * 3))
        sell_off = self.TOMATOES_SELL_OFFSET + max(0, int(-position_ratio * 3))
        buy_off = max(1, buy_off)
        sell_off = max(1, sell_off)

        if remaining_buy > 0:
            orders.append(
                Order(product, fair_int - buy_off, remaining_buy)
            )

        if remaining_sell > 0:
            orders.append(
                Order(product, fair_int + sell_off, -remaining_sell)
            )

        return orders, trader_state
