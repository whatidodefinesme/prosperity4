from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import json
import math


class Trader:
    """
    Optimized PnL trading strategy for Prosperity 4 - Round 0.

    Products:
    - EMERALDS: Stable fair value at 10000. Ultra-tight fixed-offset
      market-making with multi-level quoting.
    - TOMATOES: Linear regression predictor + tight market-making
      with contrarian market-trade signal.

    Key optimizations for 5k+ PnL:
    1. Ultra-tight spreads (±1 from fair for EMERALDS)
    2. Multi-level quoting (captures fills at different spread widths)
    3. Fixed offset from fair (not penny-jump, which widens quotes
       when book spread is large)
    4. Contrarian signal from market_trades (67% anti-predictive)
    5. Aggressive position skewing to flatten inventory
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
    ) -> Tuple[List[Order], int]:
        """
        Phase 1: Aggressively take mispriced orders from the book.
        Returns orders and updated position.
        """
        orders: List[Order] = []

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
            else:
                break

        # Hit bids above fair + edge
        for bid_price, bid_vol in sorted_buys:
            if bid_price >= fair_price + take_edge:
                sell_size = min(bid_vol, limit + position)
                if sell_size > 0:
                    orders.append(
                        Order(product, int(bid_price), -int(sell_size))
                    )
                    position -= sell_size
            else:
                break

        return orders, position

    def market_make_multilevel(
        self,
        product: str,
        fair_price: float,
        position: int,
        limit: int,
        spreads: List[Tuple[float, float]],
        skew_intensity: float,
    ) -> List[Order]:
        """
        Phase 2: Multi-level market-making with position skewing.

        Posts orders at multiple price levels from fair value.
        Each level gets a fraction of remaining capacity.
        spreads: list of (spread, volume_fraction) tuples.
        """
        orders: List[Order] = []
        if limit == 0:
            return orders

        skew = skew_intensity * (position / limit)

        buy_remaining = limit - position
        sell_remaining = limit + position

        for spread, vol_frac in spreads:
            bid_price = math.floor(fair_price - spread - skew)
            ask_price = math.ceil(fair_price + spread - skew)

            buy_qty = min(
                int(math.ceil(vol_frac * (limit - position))),
                buy_remaining,
            )
            sell_qty = min(
                int(math.ceil(vol_frac * (limit + position))),
                sell_remaining,
            )

            if buy_qty > 0:
                orders.append(
                    Order(product, int(bid_price), int(buy_qty))
                )
                buy_remaining -= buy_qty

            if sell_qty > 0:
                orders.append(
                    Order(product, int(ask_price), -int(sell_qty))
                )
                sell_remaining -= sell_qty

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
                skew_intensity = 4.0
                # Multi-level: 70% at ±1 (ultra-tight), 30% at ±2
                spreads = [(1.0, 0.7), (2.0, 0.3)]

            elif product == "TOMATOES":
                hist = trader_data["history"].get("TOMATOES", [])
                hist.append(current_mid)
                if len(hist) > self.REGRESSION_WINDOW:
                    hist.pop(0)
                trader_data["history"]["TOMATOES"] = hist
                fair_price = self.compute_regression_forecast(hist)

                # Contrarian signal from market trades:
                # bot trade direction is 67% anti-predictive
                mt = state.market_trades.get(product, [])
                if mt and len(hist) >= 2:
                    trade_signal = 0.0
                    for t in mt:
                        if t.price > current_mid:
                            trade_signal += t.quantity
                        elif t.price < current_mid:
                            trade_signal -= t.quantity
                    # Contrarian: shift fair OPPOSITE to trade direction
                    if trade_signal > 0:
                        fair_price -= 0.5
                    elif trade_signal < 0:
                        fair_price += 0.5

                take_edge = 1.0
                skew_intensity = 3.0
                # Multi-level: 70% at ±1, 30% at ±2
                spreads = [(1.0, 0.7), (2.0, 0.3)]
            else:
                fair_price = current_mid
                take_edge = 1.0
                skew_intensity = 2.0
                spreads = [(2.0, 1.0)]

            # Phase 1: Take mispriced orders
            take_orders, updated_pos = self.take_opportunities(
                product,
                fair_price,
                position,
                sorted_buys,
                sorted_sells,
                limit,
                take_edge,
            )

            # Phase 2: Multi-level market-making
            make_orders = self.market_make_multilevel(
                product,
                fair_price,
                updated_pos,
                limit,
                spreads,
                skew_intensity,
            )

            result[product] = take_orders + make_orders

        trader_data_str = json.dumps(trader_data)
        return result, 0, trader_data_str
