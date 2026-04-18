from datamodel import OrderDepth, TradingState, Order
import json
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



####
####### SYMBOLS & LIMITS #######
PEPPER_ROOT = 'INTARIAN_PEPPER_ROOT'   # steady / fixed-fair-value
OSMIUM = 'ASH_COATED_OSMIUM'           # volatile with hidden pattern

POS_LIMITS = {
    PEPPER_ROOT: 80,
    OSMIUM: 80,
}

# Parameters
MR_WINDOW = 25          # EMA window for Osmium
MR_TAKE_THRESHOLD = 1.0 # how far from EMA fair value we aggressively take liquidity

LONG, NEUTRAL, SHORT = 1, 0, -1

class ProductTrader:
    def __init__(self, name, state, prints, new_trader_data, product_group=None):
        self.orders = []
        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group
        self.last_traderData = self.get_last_traderData()
        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)
        self.expected_position = self.initial_position
        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self.get_walls()
        self.best_bid, self.best_ask = self.get_best_bid_ask()
        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self.get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self.get_total_market_buy_sell_volume()
        self.imbalance = self.get_imbalance()

    def get_imbalance(self):
        try:
            total_vol = self.total_mkt_buy_volume + self.total_mkt_sell_volume
            if total_vol == 0: return 0
            return (self.total_mkt_buy_volume - self.total_mkt_sell_volume) / total_vol
        except: pass
        return 0

    def get_last_traderData(self):
        last_traderData = {}
        try:
            if self.state.traderData != '':
                last_traderData = json.loads(self.state.traderData)
        except: pass
        return last_traderData

    def get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if len(self.mkt_buy_orders) > 0: best_bid = max(self.mkt_buy_orders.keys())
            if len(self.mkt_sell_orders) > 0: best_ask = min(self.mkt_sell_orders.keys())
        except: pass
        return best_bid, best_ask

    def get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = min([x for x,_ in self.mkt_buy_orders.items()])
        except: pass
        try: ask_wall = max([x for x,_ in self.mkt_sell_orders.items()])
        except: pass
        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
        return bid_wall, wall_mid, ask_wall

    def get_total_market_buy_sell_volume(self):
        market_bid_volume = market_ask_volume = 0
        try:
            market_bid_volume = sum([v for p, v in self.mkt_buy_orders.items()])
            market_ask_volume = sum([v for p, v in self.mkt_sell_orders.items()])
        except: pass
        return market_bid_volume, market_ask_volume

    def get_max_allowed_volume(self):
        return self.position_limit - self.initial_position, self.position_limit + self.initial_position

    def get_order_depth(self):
        buy_orders = sell_orders = {}
        try:
            order_depth: OrderDepth = self.state.order_depths[self.name]
            buy_orders = {bp: abs(bv) for bp, bv in sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
            sell_orders = {sp: abs(sv) for sp, sv in sorted(order_depth.sell_orders.items(), key=lambda x: x[0])}
        except: pass
        return buy_orders, sell_orders

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        if abs_volume > 0:
            self.orders.append(Order(self.name, int(price), abs_volume))
            if logging: self.log("BUYO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
            self.max_allowed_buy_volume -= abs_volume

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if abs_volume > 0:
            self.orders.append(Order(self.name, int(price), -abs_volume))
            if logging: self.log("SELLO", {"p":price, "s":self.name, "v":int(volume)}, product_group='ORDERS')
            self.max_allowed_sell_volume -= abs_volume

    def log(self, kind, message, product_group=None):
        if product_group is None: product_group = self.product_group
        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    def get_orders(self):
        return {self.name: self.orders}

# =============================================================================
# TRENDING PRODUCT (PEPPER ROOT)
# =============================================================================
class TrendTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(PEPPER_ROOT, state, prints, new_trader_data)
        self.fair_value = self._calculate_fair_value()

    def _calculate_fair_value(self):
        current_mid = self.wall_mid
        if current_mid is None:
            return None

        # Parametric ramp tracking
        open_mid = self.last_traderData.get(f'open_mid_{self.name}', None)
        if open_mid is None:
            open_mid = current_mid
        self.new_trader_data[f'open_mid_{self.name}'] = open_mid

        # Parametric fair based on known +1000 per 1M ms drift
        parametric_fair = open_mid + (self.state.timestamp / 1_000_000) * 1000.0

        # Short-window EMA (alpha=0.08) for local noise
        old_ema = self.last_traderData.get(f'ema_{self.name}', None)
        alpha = 0.08
        if old_ema is None:
            new_ema = current_mid
        else:
            new_ema = alpha * current_mid + (1 - alpha) * old_ema
        self.new_trader_data[f'ema_{self.name}'] = new_ema
        ema_lag_comp = 0.1 * (1 - alpha) / alpha # ~1.15 compensation for lag
        ema_fair = new_ema + ema_lag_comp

        # Blend both for robust fair value
        return (parametric_fair + ema_fair) / 2.0

    def get_orders(self):
        if self.best_bid is None or self.best_ask is None or self.fair_value is None:
            return {self.name: self.orders}

        # 1. INVENTORY SKEW
        # Bias long strongly to aggressively accumulate the 80 position limit early
        # on a parametrically drifting (trending) asset.
        long_bias = 3.0 
        inv_skew = (self.initial_position / self.position_limit) * 2.0
        adj_fair = self.fair_value - inv_skew + long_bias

        # 2. TAKING
        for sp, sv in self.mkt_sell_orders.items():
            # Aggressively take asks to ensure we hit max pos near the start
            if sp <= adj_fair + 3.0: 
                self.bid(sp, sv, logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= adj_fair + 1.5:
                self.ask(bp, bv, logging=False)

        # 3. MAKING
        bid_price = int(self.best_bid + 1)
        ask_price = int(self.best_ask - 1)
        
        if bid_price >= self.best_ask:
            bid_price = self.best_bid
        if ask_price <= self.best_bid:
            ask_price = self.best_ask

        my_bid = min(bid_price, int(adj_fair - 2.0))
        my_ask = max(ask_price, int(adj_fair + 2.0))

        if my_bid > 0:
            self.bid(my_bid, self.max_allowed_buy_volume)
        if my_ask > 0:
            self.ask(my_ask, self.max_allowed_sell_volume)

        return {self.name: self.orders}

# =============================================================================
# VOLATILE PRODUCT (OSMIUM)
# =============================================================================
class OsmiumTrader(ProductTrader):
    def __init__(self, state, prints, new_trader_data):
        super().__init__(OSMIUM, state, prints, new_trader_data)
        self.fair_value = self._calculate_fair_value()

    def _calculate_fair_value(self):
        current_mid = self.wall_mid
        if current_mid is None:
            return 10000.0

        old_ema = self.last_traderData.get(f'ema_{self.name}', None)
        alpha = 2.0 / (MR_WINDOW + 1)
        if old_ema is None:
            new_ema = current_mid
        else:
            new_ema = alpha * current_mid + (1 - alpha) * old_ema
            
        self.new_trader_data[f'ema_{self.name}'] = new_ema
        return new_ema

    def get_orders(self):
        if self.best_bid is None or self.best_ask is None or self.fair_value is None:
            return {self.name: self.orders}

        # 1. INVENTORY SKEW
        # Strong mean reversion -> reset around fair
        inv_skew = (self.initial_position / self.position_limit) * 4.0
        adj_fair = self.fair_value - inv_skew

        # 2. TAKING
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= adj_fair - 1.5:
                self.bid(sp, sv, logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= adj_fair + 1.5:
                self.ask(bp, bv, logging=False)

        # 3. MAKING
        # Penny jump best inside the spread
        bid_price = int(self.best_bid + 1)
        ask_price = int(self.best_ask - 1)
        
        if bid_price >= self.best_ask:
            bid_price = self.best_bid
        if ask_price <= self.best_bid:
            ask_price = self.best_ask

        # Hard cap quotes inside fair value constraint to avoid getting run over
        my_bid = min(bid_price, int(adj_fair - 1.0))
        my_ask = max(ask_price, int(adj_fair + 1.0))

        if my_bid > 0:
            self.bid(my_bid, self.max_allowed_buy_volume)
        if my_ask > 0:
            self.ask(my_ask, self.max_allowed_sell_volume)

        return {self.name: self.orders}

# =============================================================================
# MAIN TRADER
# =============================================================================
class Trader:
    # Position limits — also referenced by our local backtester
    POSITION_LIMITS = {
        "EMERALDS": 20,
        "TOMATOES": 20,
        "INTARIAN_PEPPER_ROOT": 80,
        "ASH_COATED_OSMIUM": 80,
    }

    def run(self, state: TradingState):
        result: dict[str, list[Order]] = {}
        conversions =0
        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position
            },
        }

        product_traders = {
            PEPPER_ROOT: TrendTrader,      
            OSMIUM: OsmiumTrader,           
        }

        for symbol, TraderClass in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = TraderClass(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except Exception as e:
                    pass  # safety – never crash the bot

        try:
            final_trader_data = json.dumps(new_trader_data)
        except:
            final_trader_data = ''

        # try:
        #     print(json.dumps(prints))
        # except:
        #     pass
        logger.flush(state, result, conversions, final_trader_data)
        return result, conversions, final_trader_data