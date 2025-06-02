# trade_generator.py
from datetime import datetime, timedelta
import random


class Trade:
    def __init__(self, symbol, timestamp, order_type, price, size, direction, holding_period, exit_strategy, meta):
        self.symbol = symbol
        self.timestamp = timestamp
        self.order_type = order_type
        self.price = price
        self.size = size
        self.direction = direction
        self.holding_period = holding_period
        self.exit_strategy = exit_strategy
        self.exit_time = timestamp + holding_period
        self.meta = meta


# --- RANDOM ---
def generate_random_trades(stocks, date, vol_regime):
    trades = []
    for symbol in stocks:
        price = 100  # placeholder; replace with actual price fetch
        trade = Trade(
            symbol=symbol,
            timestamp=datetime.combine(date, datetime.min.time()),
            order_type="MARKET",
            price=price,
            size=round(random.uniform(0.1, 1.0), 2),
            direction=random.choice([1, -1]),
            holding_period=timedelta(days=1),
            exit_strategy="TIME_BASED",
            meta={"exploration_type": "random", "novelty": random.uniform(0.3, 1.0)}
        )
        trades.append(trade)
    return trades


# --- RULE BASED ---
def generate_rule_based_trades(stocks, date, vol_regime):
    trades = []
    for symbol in stocks:
        # Example heuristic: Buy if pseudo RSI < 30
        pseudo_rsi = random.uniform(10, 70)
        if pseudo_rsi < 30:
            price = 100  # placeholder
            trade = Trade(
                symbol=symbol,
                timestamp=datetime.combine(date, datetime.min.time()),
                order_type="MARKET",
                price=price,
                size=0.3,
                direction=1,
                holding_period=timedelta(days=2),
                exit_strategy="TIME_BASED",
                meta={"exploration_type": "rule_based", "rsi": pseudo_rsi, "novelty": 0.7}
            )
            trades.append(trade)
    return trades


# --- MODEL BASED ---
def generate_model_based_trades(stocks, date, vol_regime):
    trades = []
    for symbol in stocks:
        price = 100  # placeholder
        model_output = {
            "entry": True,
            "direction": 1,
            "size": 0.4,
            "holding_days": 2,
            "confidence": 0.85
        }
        if model_output["entry"]:
            trade = Trade(
                symbol=symbol,
                timestamp=datetime.combine(date, datetime.min.time()),
                order_type="MARKET",
                price=price,
                size=model_output["size"],
                direction=model_output["direction"],
                holding_period=timedelta(days=model_output["holding_days"]),
                exit_strategy="TIME_BASED",
                meta={"exploration_type": "model", "confidence": model_output["confidence"], "novelty": 0.6}
            )
            trades.append(trade)
    return trades
