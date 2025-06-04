# trade_generator.py

from datetime import datetime, timedelta
import random
from core.logger.logger import logger
from core.data_provider.data_provider import get_last_close


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


def generate_random_trades(stocks, date, vol_regime):
    from core.data_provider.data_provider import get_last_close
    trades = []

    logger.warning(f"⚡ generate_random_trades() called for {date} with {len(stocks)} stocks: {stocks}")

    for symbol in stocks:
        logger.debug(f"🔍 Attempting trade for {symbol}")
        price = get_last_close(symbol, sim_date=date)

        if price is None:
            logger.warning(f"❌ Skipping {symbol}: get_last_close() returned None.")
            continue

        logger.debug(f"✅ {symbol}: Price found = {price}")

        trade = Trade(
            symbol=symbol,
            timestamp=datetime.combine(date, datetime.min.time()),
            order_type="MARKET",
            price=price,
            size=round(random.uniform(0.1, 1.0), 2),
            direction=random.choice([1, -1]),
            holding_period=timedelta(days=1),
            exit_strategy="TIME_BASED",
            meta={
                "exploration_type": "random",
                "novelty": random.uniform(0.3, 1.0),
                "interval": "day",
                "strategy_config": {},
                "sma_short": 10,
                "sma_long": 30,
                "rsi_thresh": 40,
                "confidence": round(random.uniform(0.4, 0.9), 2),
                "sharpe": round(random.uniform(0.0, 2.0), 2),
                "rank": random.randint(1, 100)
            }
        )
        trades.append(trade)

    logger.warning(f"📦 {len(trades)} random trades generated on {date}")
    return trades


def generate_rule_based_trades(stocks, date, vol_regime):
    trades = []
    for symbol in stocks:
        pseudo_rsi = random.uniform(10, 70)
        if pseudo_rsi < 30:
            price = get_last_close(symbol, sim_date=date)
            if price is None:
                logger.warning(f"❌ Skipping {symbol}: No price data available.")
                continue

            trade = Trade(
                symbol=symbol,
                timestamp=datetime.combine(date, datetime.min.time()),
                order_type="MARKET",
                price=price,
                size=0.3,
                direction=1,
                holding_period=timedelta(days=2),
                exit_strategy="TIME_BASED",
                meta={
                    "exploration_type": "rule_based",
                    "rsi": pseudo_rsi,
                    "novelty": 0.7,
                    "interval": "day",
                    "strategy_config": {},
                    "sma_short": 10,
                    "sma_long": 30,
                    "rsi_thresh": pseudo_rsi,
                    "confidence": 0.75,
                    "sharpe": 1.1,
                    "rank": random.randint(1, 50)
                }
            )
            trades.append(trade)
    logger.info(f"🧠 Generated {len(trades)} rule-based trades on {date}")
    return trades


def generate_model_based_trades(stocks, date, vol_regime):
    trades = []
    for symbol in stocks:
        price = get_last_close(symbol)
        if price is None:
            logger.warning(f"❌ Skipping {symbol}: No price data available.")
            continue

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
                meta={
                    "exploration_type": "model",
                    "confidence": model_output["confidence"],
                    "novelty": 0.6,
                    "interval": "day",
                    "strategy_config": {},
                    "sma_short": 10,
                    "sma_long": 30,
                    "rsi_thresh": 40,
                    "sharpe": 1.3,
                    "rank": random.randint(1, 30)
                }
            )
            trades.append(trade)
    logger.info(f"🤖 Generated {len(trades)} model-based trades on {date}")
    return trades
