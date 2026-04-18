import json
import math
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
# Strategy:
#
#   ASH_COATED_OSMIUM (Round 2)
#     - Pinned at fair = 10000 on every day observed (std ~5, spread ~16).
#     - Lag-1 autocorrelation of Δmid ≈ -0.50 → strong bid-ask bounce.
#     - Behaviour is essentially identical to EMERALDS / AMETHYSTS from prior
#       rounds. Fixed fair + penny-jump market-making + take mispriced book
#       levels is the right approach.
#
#   INTARIAN_PEPPER_ROOT (Round 2)
#     - Deterministic +1000 / day drift (11000 → 12000 → 13000 → 14000 across
#       the three observed days), std ~288 per day, spread ~13–15.
#     - Lag-1 AC of Δmid ≈ -0.50 as well — tick-level mean-revert sits on top
#       of the low-frequency ramp.
#     - We therefore use a rolling linear-regression forecast of the mid as
#       fair (captures both the trend and the bounce) and market-make around
#       it. This mirrors the proven TOMATOES structure.
#
#   EMERALDS / TOMATOES (Round 0)
#     - Kept for backwards compatibility with the local backtester's Round 0
#       CSVs; same config that previously produced the best PnL.
#
# For each product we run a two-phase loop:
#   Phase 1 (take): cross the book whenever we can lock in ≥ `take_edge`
#                   ticks of edge relative to our fair price.
#   Phase 2 (make): post a penny-jumping bid/ask anchored to the reconciled
#                   post-take book, but never worse than fair ± make_spread.
# ---------------------------------------------------------------------------


class Trader:
    POSITION_LIMITS = {
        # Round 0 products kept for backtester compatibility.
        "EMERALDS": 80,
        "TOMATOES": 80,
        # Round 2 products.
        "ASH_COATED_OSMIUM": 50,
        "INTARIAN_PEPPER_ROOT": 50,
    }

    # Products with a hard-coded fair value.
    STABLE_FAIRS = {
        "EMERALDS": 10000.0,
        "ASH_COATED_OSMIUM": 10000.0,
    }

    # Products whose fair is computed from a rolling mid-price history.
    TREND_PRODUCTS = ("TOMATOES", "INTARIAN_PEPPER_ROOT")

    # Minimum edge vs fair to cross the book.
    TAKE_EDGE = {
        "EMERALDS": 1.0,
        "TOMATOES": 1.5,
        "ASH_COATED_OSMIUM": 1.0,
        "INTARIAN_PEPPER_ROOT": 2.0,
    }

    # Target half-spread around fair for passive market-making quotes.
    MAKE_SPREAD = {
        "EMERALDS": 2.5,
        "TOMATOES": 1.5,
        "ASH_COATED_OSMIUM": 2.0,
        "INTARIAN_PEPPER_ROOT": 2.5,
    }

    # Inventory skew (how many ticks we shift both quotes when fully long/short).
    SKEW_INTENSITY = {
        "EMERALDS": 3.0,
        "TOMATOES": 2.0,
        "ASH_COATED_OSMIUM": 3.0,
        "INTARIAN_PEPPER_ROOT": 2.0,
    }

    # Rolling window for the linear-regression forecast used on trend products.
    REGRESSION_WINDOW = 15

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mid(sorted_buys: list[tuple[int, int]], sorted_sells: list[tuple[int, int]]) -> float | None:
        if not sorted_buys or not sorted_sells:
            return None
        return (sorted_buys[0][0] + sorted_sells[0][0]) / 2.0

    @staticmethod
    def _regression_forecast(history: list[float]) -> float:
        """One-step-ahead forecast from a simple OLS line fit to `history`."""
        n = len(history)
        if n < 2:
            return history[-1]

        sum_x = (n * (n - 1)) / 2.0
        sum_x_sq = (n * (n - 1) * (2 * n - 1)) / 6.0
        sum_y = sum(history)
        sum_xy = sum(i * y for i, y in enumerate(history))

        mean_x = sum_x / n
        mean_y = sum_y / n

        denom = sum_x_sq - n * (mean_x ** 2)
        if denom == 0:
            return history[-1]

        slope = (sum_xy - n * mean_x * mean_y) / denom
        intercept = mean_y - slope * mean_x
        return slope * n + intercept

    # ------------------------------------------------------------------
    # Phase 1: take mispriced orders in the book
    # ------------------------------------------------------------------
    def _take(
        self,
        product: str,
        fair: float,
        position: int,
        sorted_buys: list[tuple[int, int]],
        sorted_sells: list[tuple[int, int]],
        limit: int,
    ) -> tuple[list[Order], int, int | None, int | None]:
        orders: list[Order] = []
        edge = self.TAKE_EDGE.get(product, 1.0)

        best_remaining_ask: int | None = sorted_sells[0][0] if sorted_sells else None
        best_remaining_bid: int | None = sorted_buys[0][0] if sorted_buys else None

        # Buy side: sweep asks priced at or below fair - edge.
        for ask_price, ask_vol in sorted_sells:
            if ask_price <= fair - edge:
                available = abs(ask_vol)
                size = min(available, limit - position)
                if size > 0:
                    orders.append(Order(product, int(ask_price), int(size)))
                    position += size
                if size < available:
                    best_remaining_ask = ask_price
                    break
            else:
                best_remaining_ask = ask_price
                break
        else:
            best_remaining_ask = None

        # Sell side: sweep bids priced at or above fair + edge.
        for bid_price, bid_vol in sorted_buys:
            if bid_price >= fair + edge:
                available = bid_vol
                size = min(available, limit + position)
                if size > 0:
                    orders.append(Order(product, int(bid_price), -int(size)))
                    position -= size
                if size < available:
                    best_remaining_bid = bid_price
                    break
            else:
                best_remaining_bid = bid_price
                break
        else:
            best_remaining_bid = None

        return orders, position, best_remaining_bid, best_remaining_ask

    # ------------------------------------------------------------------
    # Phase 2: passive penny-jumped market-making quotes
    # ------------------------------------------------------------------
    def _make(
        self,
        product: str,
        fair: float,
        position: int,
        best_remaining_bid: int | None,
        best_remaining_ask: int | None,
        limit: int,
    ) -> list[Order]:
        orders: list[Order] = []
        if limit == 0:
            return orders

        base_spread = self.MAKE_SPREAD.get(product, 2.0)
        skew_intensity = self.SKEW_INTENSITY.get(product, 2.0)
        # Skew pushes both quotes *away* from the side we're already loaded on.
        skew = skew_intensity * (position / limit)

        ideal_bid = math.floor(fair - base_spread - skew)
        ideal_ask = math.ceil(fair + base_spread - skew)

        market_bid = best_remaining_bid if best_remaining_bid is not None else ideal_bid
        market_ask = best_remaining_ask if best_remaining_ask is not None else ideal_ask

        # Penny-jump: sit one tick inside the best remaining market quote,
        # but never worse than our ideal (protects edge).
        optimal_bid = min(ideal_bid, market_bid + 1)
        optimal_ask = max(ideal_ask, market_ask - 1)

        buy_size = limit - position
        sell_size = limit + position

        if buy_size > 0:
            orders.append(Order(product, int(optimal_bid), int(buy_size)))
        if sell_size > 0:
            orders.append(Order(product, int(optimal_ask), -int(sell_size)))

        return orders

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result: dict[Symbol, list[Order]] = {}
        conversions = 0

        try:
            memory = json.loads(state.traderData) if state.traderData else {}
        except (json.JSONDecodeError, TypeError):
            memory = {}
        history: dict[str, list[float]] = memory.get("history", {})

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = self.POSITION_LIMITS.get(product, 20)

            sorted_sells = sorted(order_depth.sell_orders.items())
            sorted_buys = sorted(order_depth.buy_orders.items(), reverse=True)

            current_mid = self._mid(sorted_buys, sorted_sells)
            if current_mid is None:
                # Empty book on one side — skip this tick for this product.
                continue

            # --- Fair-value estimation ---
            if product in self.STABLE_FAIRS:
                fair_price = self.STABLE_FAIRS[product]
            elif product in self.TREND_PRODUCTS:
                hist = history.get(product, [])
                hist.append(current_mid)
                if len(hist) > self.REGRESSION_WINDOW:
                    hist.pop(0)
                history[product] = hist
                fair_price = self._regression_forecast(hist)
            else:
                # Unknown product — fall back to mid; still safe to market-make.
                fair_price = current_mid

            # --- Phase 1: take ---
            take_orders, position_after_take, best_rem_bid, best_rem_ask = self._take(
                product, fair_price, position, sorted_buys, sorted_sells, limit
            )

            # --- Phase 2: make ---
            make_orders = self._make(
                product, fair_price, position_after_take, best_rem_bid, best_rem_ask, limit
            )

            orders = take_orders + make_orders
            if orders:
                result[product] = orders

        memory["history"] = history
        trader_data = json.dumps(memory, separators=(",", ":"))

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
