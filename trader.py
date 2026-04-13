from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json
import math


class Trader:
    """
    Optimized PnL trading strategy for Prosperity 4 - Round 0.

    Products:
    - EMERALDS: Stable fair value at 10000. Tight market-making with
      penny-jump anchoring and position-proportional skewing.
    - TOMATOES: Linear regression predictor + tight market-making.

    Key insight: virtually all PnL comes from passive fills (bots trading
    against our limit orders). Strategy maximizes fill probability via:
    1. Tight spreads (close to fair value)
    2. Penny-jump anchoring (always tightest quote in book)
    3. Aggressive position skewing (flatten inventory fast)
    4. Reconciled book state (anchor to post-take liquidity)
    """

    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    REGRESSION_WINDOW = 15

    def bid(self):
        return 15

    def compute_regression_forecast(self, history: List[float]) -> float:
        """Linear regression on price history to predict next value."""
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
        take_edge: float,
    ) -> Tuple[List[Order], int, int, int]:
        """
        Phase 1: Aggressively take mispriced orders from the book.

        Returns orders, updated position, and the remaining best bid/ask
        after our takes (for anchoring market-making quotes).
        """
        orders: List[Order] = []

        best_remaining_ask = sorted_sells[0][0] if sorted_sells else None
        best_remaining_bid = sorted_buys[0][0] if sorted_buys else None

        # Hit asks below fair - edge
        for ask_price, ask_vol in sorted_sells:
            if ask_price <= fair_price - take_edge:
                available = abs(ask_vol)
                buy_size = min(available, limit - position)
                if buy_size > 0:
                    orders.append(
                        Order(product, int(ask_price), int(buy_size))
                    )
                    position += buy_size
                if buy_size < available:
                    best_remaining_ask = ask_price
                    break
            else:
                best_remaining_ask = ask_price
                break
        else:
            best_remaining_ask = None

        # Hit bids above fair + edge
        for bid_price, bid_vol in sorted_buys:
            if bid_price >= fair_price + take_edge:
                sell_size = min(bid_vol, limit + position)
                if sell_size > 0:
                    orders.append(
                        Order(product, int(bid_price), -int(sell_size))
                    )
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
        base_spread: float,
        skew_intensity: float,
    ) -> List[Order]:
        """
        Phase 2: Post market-making quotes anchored to reconciled book.

        Uses penny-jumping (best_remaining_bid+1 / best_remaining_ask-1)
        constrained by ideal spread from fair value. Position-proportional
        skewing shifts both bid and ask to flatten inventory.
        """
        orders: List[Order] = []
        if limit == 0:
            return orders

        skew = skew_intensity * (position / limit)

        ideal_bid = math.floor(fair_price - base_spread - skew)
        ideal_ask = math.ceil(fair_price + base_spread - skew)

        # Anchor to remaining book liquidity (penny-jump)
        best_market_bid = (
            best_remaining_bid
            if best_remaining_bid is not None
            else ideal_bid
        )
        best_market_ask = (
            best_remaining_ask
            if best_remaining_ask is not None
            else ideal_ask
        )

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
        result: Dict[str, List[Order]] = {}

        # Deserialize trader data
        trader_data = {}
        if state.traderData and state.traderData != "":
            try:
                trader_data = json.loads(state.traderData)
            except (json.JSONDecodeError, TypeError):
                trader_data = {}

        if "history" not in trader_data:
            trader_data["history"] = {"TOMATOES": []}

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            sorted_sells = sorted(order_depth.sell_orders.items())
            sorted_buys = sorted(
                order_depth.buy_orders.items(), reverse=True
            )

            if not sorted_buys or not sorted_sells:
                result[product] = []
                continue

            current_mid = (sorted_buys[0][0] + sorted_sells[0][0]) / 2.0

            if product == "EMERALDS":
                fair_price = 10000.0
                take_edge = 1.0
                base_spread = 2.0
                skew_intensity = 3.0
            elif product == "TOMATOES":
                hist = trader_data["history"].get("TOMATOES", [])
                hist.append(current_mid)
                if len(hist) > self.REGRESSION_WINDOW:
                    hist.pop(0)
                trader_data["history"]["TOMATOES"] = hist
                fair_price = self.compute_regression_forecast(hist)
                take_edge = 1.0
                base_spread = 1.5
                skew_intensity = 2.0
            else:
                fair_price = current_mid
                take_edge = 1.0
                base_spread = 2.0
                skew_intensity = 2.0

            # Phase 1: Take mispriced orders
            take_orders, updated_pos, rem_bid, rem_ask = (
                self.take_opportunities(
                    product,
                    fair_price,
                    position,
                    sorted_buys,
                    sorted_sells,
                    limit,
                    take_edge,
                )
            )

            # Phase 2: Market-make with reconciled book
            make_orders = self.market_make(
                product,
                fair_price,
                updated_pos,
                rem_bid,
                rem_ask,
                limit,
                base_spread,
                skew_intensity,
            )

            result[product] = take_orders + make_orders

        trader_data_str = json.dumps(trader_data)
        return result, 0, trader_data_str
