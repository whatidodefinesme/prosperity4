from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple
import json
import math


class Trader:
    """
    Optimized trading strategy for Prosperity 4 - Round 0.

    Restores the proven penny-jump anchoring structure that achieved 2.5k PnL,
    with conservative data-driven improvements:
    1. Regression window 15->10 (lower prediction error: 0.938 vs 0.954)
    2. Slightly tighter spreads for rare narrow-book moments
    3. Reconciled book state (anchor to post-take liquidity)
    4. json instead of jsonpickle (no extra dependency)

    Core insight: penny-jump anchoring (best_bid+1 / best_ask-1) gives
    ~7 ticks edge for EMERALDS and ~5.6 ticks for TOMATOES. The min/max
    constraint ensures we never bid above ideal or ask below ideal.
    """

    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    def __init__(self):
        self.regression_window = 10
        self.take_edge = {"EMERALDS": 1.0, "TOMATOES": 1.5}
        self.make_spread = {"EMERALDS": 2.5, "TOMATOES": 1.5}

    def bid(self):
        return 15

    def get_mid_price(
        self,
        sorted_buys: List[Tuple[int, int]],
        sorted_sells: List[Tuple[int, int]],
    ) -> float:
        if not sorted_buys or not sorted_sells:
            return None
        return (sorted_buys[0][0] + sorted_sells[0][0]) / 2.0

    def compute_regression_forecast(self, history: List[float]) -> float:
        n = len(history)
        if n < 2:
            return history[-1]

        sum_x = (n * (n - 1)) / 2
        sum_x_sq = (n * (n - 1) * (2 * n - 1)) / 6

        sum_y = sum(history)
        sum_xy = sum(i * y for i, y in enumerate(history))

        mean_x = sum_x / n
        mean_y = sum_y / n

        denominator = sum_x_sq - n * (mean_x ** 2)
        if denominator == 0:
            return history[-1]

        slope = (sum_xy - n * mean_x * mean_y) / denominator
        intercept = mean_y - slope * mean_x

        return slope * n + intercept

    def take_opportunities(
        self,
        product: str,
        fair_price: float,
        position: int,
        sorted_buys: List[Tuple[int, int]],
        sorted_sells: List[Tuple[int, int]],
        limit: int,
    ) -> Tuple[List[Order], int, int, int]:
        """
        Phase 1: Take mispriced orders from the book.

        Returns orders, updated position, and remaining best bid/ask
        after our takes (for anchoring market-making quotes).
        """
        orders = []
        edge = self.take_edge.get(product, 1.0)

        best_remaining_ask = sorted_sells[0][0] if sorted_sells else None
        best_remaining_bid = sorted_buys[0][0] if sorted_buys else None

        for ask_price, ask_vol in sorted_sells:
            if ask_price <= fair_price - edge:
                available_vol = abs(ask_vol)
                buy_size = min(available_vol, limit - position)

                if buy_size > 0:
                    orders.append(Order(product, int(ask_price), int(buy_size)))
                    position += buy_size

                if buy_size < available_vol:
                    best_remaining_ask = ask_price
                    break
            else:
                best_remaining_ask = ask_price
                break
        else:
            best_remaining_ask = None

        for bid_price, bid_vol in sorted_buys:
            if bid_price >= fair_price + edge:
                sell_size = min(bid_vol, limit + position)

                if sell_size > 0:
                    orders.append(Order(product, int(bid_price), -int(sell_size)))
                    position -= sell_size

                if sell_size < bid_vol:
                    best_remaining_bid = bid_price
                    break
            else:
                best_remaining_bid = bid_price
                break
        else:
            best_remaining_bid = None

        return orders, position, best_remaining_bid, best_remaining_ask

    def market_make(
        self,
        product: str,
        fair_price: float,
        position: int,
        best_remaining_bid: int,
        best_remaining_ask: int,
        limit: int,
    ) -> List[Order]:
        """
        Phase 2: Post market-making quotes anchored to reconciled book.

        Uses penny-jumping (best_remaining_bid+1 / best_remaining_ask-1)
        constrained by ideal spread from fair value. The min/max ensures
        we never bid above (fair - spread) or ask below (fair + spread).

        Position-proportional skewing shifts both bid and ask to flatten
        inventory faster.
        """
        orders = []
        if limit == 0:
            return orders

        skew_intensity = 3.0 if product == "EMERALDS" else 2.0
        skew = skew_intensity * (position / limit)
        base_spread = self.make_spread.get(product, 2.0)

        ideal_bid = math.floor(fair_price - base_spread - skew)
        ideal_ask = math.ceil(fair_price + base_spread - skew)

        best_market_bid = (
            best_remaining_bid if best_remaining_bid is not None else ideal_bid
        )
        best_market_ask = (
            best_remaining_ask if best_remaining_ask is not None else ideal_ask
        )

        # Penny-jump: be the tightest quote in the book
        # min() ensures we never bid higher than our ideal (protects edge)
        # max() ensures we never ask lower than our ideal (protects edge)
        optimal_bid = min(ideal_bid, best_market_bid + 1)
        optimal_ask = max(ideal_ask, best_market_ask - 1)

        buy_size = limit - position
        sell_size = limit + position

        if buy_size > 0:
            orders.append(Order(product, int(optimal_bid), int(buy_size)))
        if sell_size > 0:
            orders.append(Order(product, int(optimal_ask), -int(sell_size)))

        return orders

    def run(self, state: TradingState):
        result = {}

        try:
            trader_data = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            trader_data = {}

        if "history" not in trader_data:
            trader_data["history"] = {"TOMATOES": []}

        for product in state.order_depths.keys():
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            sorted_sells = sorted(order_depth.sell_orders.items())
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)

            product_orders = []

            current_mid = self.get_mid_price(sorted_buys, sorted_sells)
            if current_mid is None:
                continue

            if product == "EMERALDS":
                fair_price = 10000.0

            elif product == "TOMATOES":
                hist = trader_data["history"].get("TOMATOES", [])
                hist.append(current_mid)

                if len(hist) > self.regression_window:
                    hist.pop(0)

                trader_data["history"]["TOMATOES"] = hist
                fair_price = self.compute_regression_forecast(hist)
            else:
                fair_price = current_mid

            take_orders, updated_position, best_rem_bid, best_rem_ask = (
                self.take_opportunities(
                    product, fair_price, position, sorted_buys, sorted_sells, limit
                )
            )
            product_orders.extend(take_orders)

            make_orders = self.market_make(
                product,
                fair_price,
                updated_position,
                best_rem_bid,
                best_rem_ask,
                limit,
            )
            product_orders.extend(make_orders)

            result[product] = product_orders

        traderData = json.dumps(trader_data)
        return result, 0, traderData
