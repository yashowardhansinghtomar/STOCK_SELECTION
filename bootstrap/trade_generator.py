# bootstrap/trade_generator.py

from datetime import datetime, timedelta
import random
from core.logger.logger import logger
from core.data_provider.data_provider import get_last_close, fetch_stock_data
from utils.time_utils import make_naive  # ✅ added for consistency


class Trade:
    def __init__(self, symbol, timestamp, order_type, price, size, direction, holding_period, exit_strategy, meta):
        self.symbol = symbol
        self.timestamp = make_naive(timestamp)  # ✅ enforce naive
        self.order_type = order_type
        self.price = price
        self.size = size
        self.direction = direction
        self.holding_period = holding_period
        self.exit_strategy = exit_strategy
        self.exit_time = self.timestamp + holding_period  # remains naive
        self.meta = meta


def generate_random_trades(stocks, date, vol_regime):
    trades = []
    skipped = []

    logger.warning(f"⚡ generate_random_trades() called for {date} | Total: {len(stocks)} | Sample: {stocks[:5]} ...")

    for symbol in stocks:
        logger.debug(f"🔍 Attempting trade for {symbol}")

        # Retry logic to refetch if price is missing on first try
        price = get_last_close(symbol, sim_date=date)
        if price is None:
            logger.info(f"🔁 Attempting to refetch {symbol} minute data for {date}...")
            _ = fetch_stock_data(symbol, interval="minute", start=date - timedelta(days=1), end=date + timedelta(days=1))
            price = get_last_close(symbol, sim_date=date)

        if price is None:
            logger.warning(f"❌ Skipping {symbol}: No valid minute data on {date}")
            skipped.append(symbol)
            continue

        logger.debug(f"✅ {symbol}: Price found = {price}")

        # 50% chance intraday or multiday
        if random.random() < 0.5:
            interval = "15minute"
            holding_period = timedelta(minutes=random.choice([30, 60, 90]))
        else:
            interval = "day"
            holding_period = timedelta(days=random.choice([1, 2, 3]))

        trade = Trade(
            symbol=symbol,
            timestamp=datetime.combine(date, datetime.min.time()),
            order_type="MARKET",
            price=price,
            size=round(random.uniform(0.1, 1.0), 2),
            direction=random.choice([1, -1]),
            holding_period=holding_period,
            exit_strategy="TIME_BASED",
            meta={
                "exploration_type": "random",
                "novelty": random.uniform(0.3, 1.0),
                "interval": interval,
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
    if skipped:
        logger.warning(f"🚫 {len(skipped)} stocks skipped due to missing data: {skipped[:10]}...")
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

            if random.random() < 0.5:
                interval = "15minute"
                holding_period = timedelta(minutes=random.choice([30, 60, 90]))
            else:
                interval = "day"
                holding_period = timedelta(days=random.choice([2, 3]))

            trade = Trade(
                symbol=symbol,
                timestamp=datetime.combine(date, datetime.min.time()),
                order_type="MARKET",
                price=price,
                size=0.3,
                direction=1,
                holding_period=holding_period,
                exit_strategy="TIME_BASED",
                meta={
                    "exploration_type": "rule_based",
                    "rsi": pseudo_rsi,
                    "novelty": 0.7,
                    "interval": interval,
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
    logger.info(f"🧐 Generated {len(trades)} rule-based trades on {date}")
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
            "confidence": 0.85
        }
        if model_output["entry"]:
            if random.random() < 0.5:
                interval = "15minute"
                holding_period = timedelta(minutes=random.choice([30, 60, 90]))
            else:
                interval = "day"
                holding_period = timedelta(days=random.choice([2, 3]))

            trade = Trade(
                symbol=symbol,
                timestamp=datetime.combine(date, datetime.min.time()),
                order_type="MARKET",
                price=price,
                size=model_output["size"],
                direction=model_output["direction"],
                holding_period=holding_period,
                exit_strategy="TIME_BASED",
                meta={
                    "exploration_type": "model",
                    "confidence": model_output["confidence"],
                    "novelty": 0.6,
                    "interval": interval,
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