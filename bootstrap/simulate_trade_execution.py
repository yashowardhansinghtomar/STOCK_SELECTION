# bootstrap/simulate_trade_execution.py

from core.data_provider.data_provider import fetch_stock_data
from core.realism_boosters.slippage import calculate_slippage
from core.realism_boosters.market_impact import estimate_market_impact
from core.logger.logger import logger
from datetime import timedelta, datetime
import random
import pandas as pd
from utils.time_utils import to_naive_utc, make_naive

def simulate_trade_execution(trade, date):
    # Force full session fetch from 02:45 to 10:00 UTC
    start_time = datetime.combine(date, datetime.strptime("02:45", "%H:%M").time())
    end_time = datetime.combine(date, datetime.strptime("10:00", "%H:%M").time())
    bars_df = fetch_stock_data(trade.symbol, interval="minute", start=start_time, end=end_time)

    if bars_df is None or bars_df.empty:
        logger.warning(f"❌ No price data available for {trade.symbol} on {date}")
        return None

    try:
        if bars_df.index.name in ("date", "timestamp"):
            bars_df = bars_df.reset_index().rename(columns={bars_df.index.name: "timestamp"})
        elif "date" in bars_df.columns:
            bars_df = bars_df.rename(columns={"date": "timestamp"})

        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"])
        bars_df = to_naive_utc(bars_df, "timestamp")

        if trade.order_type == "MARKET":
            exe_time = make_naive(trade.timestamp + timedelta(minutes=random.randint(1, 5)))
            market_open_utc = datetime.combine(exe_time.date(), datetime.strptime("03:45", "%H:%M").time())
            if exe_time < market_open_utc:
                exe_time = market_open_utc

            row = bars_df[bars_df["timestamp"] == exe_time]
            if row.empty:
                window = bars_df[(bars_df["timestamp"] >= exe_time) & (bars_df["timestamp"] <= exe_time + timedelta(minutes=5))]
                if window.empty:
                    # fallback: look backwards
                    fallback_window = bars_df[
                        (bars_df["timestamp"] < exe_time) &
                        (bars_df["timestamp"] >= exe_time - timedelta(minutes=15))
                    ].sort_values("timestamp", ascending=False)

                    if fallback_window.empty:
                        logger.warning(f"❌ No fallback candle for {trade.symbol} after or before {exe_time}")
                        return None

                    row = fallback_window.iloc[0:1]

            exe_price = row.iloc[0]["open"]
            slippage = calculate_slippage(trade.symbol, trade.size, date)
            impact = estimate_market_impact(trade.size, trade.symbol, date)
            exe_price *= (1 + slippage + impact)

        elif trade.order_type == "LIMIT":
            fill_prob = trade.meta.get("fill_prob", 0.7)
            if random.random() < fill_prob:
                exe_price = trade.price
            else:
                return None

        if trade.exit_strategy == "TIME_BASED":
            if trade.meta.get("interval") == "15minute":
                latest_exit = datetime.combine(trade.timestamp.date(), datetime.strptime("15:15", "%H:%M").time())
                if trade.exit_time > latest_exit:
                    logger.info(f"⏰ Clamping exit_time for {trade.symbol} to 3:15 PM")
                    trade.exit_time = latest_exit

            exit_date = trade.exit_time.date()
            exit_start = datetime.combine(exit_date, datetime.strptime("02:45", "%H:%M").time())
            exit_end = datetime.combine(exit_date, datetime.strptime("10:00", "%H:%M").time())
            exit_bars_df = fetch_stock_data(trade.symbol, interval="minute", start=exit_start, end=exit_end)
            if exit_bars_df is None or exit_bars_df.empty:
                logger.warning(f"❌ No exit data for {trade.symbol} on {exit_date}")
                return None

            if exit_bars_df.index.name in ("date", "timestamp"):
                exit_bars_df = exit_bars_df.reset_index().rename(columns={exit_bars_df.index.name: "timestamp"})
            elif "date" in exit_bars_df.columns:
                exit_bars_df = exit_bars_df.rename(columns={"date": "timestamp"})

            exit_bars_df["timestamp"] = pd.to_datetime(exit_bars_df["timestamp"])
            exit_bars_df = to_naive_utc(exit_bars_df, "timestamp")

            exit_time = make_naive(trade.exit_time)
            exit_row = exit_bars_df[exit_bars_df["timestamp"] == exit_time]
            if exit_row.empty:
                logger.warning(f"❌ No exit candle for {trade.symbol} at {exit_time} — using entry price.")
            exit_price = exe_price if exit_row.empty else exit_row.iloc[0]["close"]

        reward = calculate_reward(entry=exe_price, exit=exit_price, trade=trade)

        result = {
            "symbol": trade.symbol,
            "entry_price": exe_price,
            "exit_price": exit_price,
            "entry_time": exe_time,
            "exit_time": trade.exit_time,
            "reward": reward,
            "direction": trade.direction,
            "meta": trade.meta
        }

        logger.info(f"🎯 Simulated trade: {trade.symbol} | Entry={exe_price:.2f}, Exit={exit_price:.2f}, Reward={reward:.4f}")
        return result

    except Exception as e:
        logger.warning(f"❌ Simulation failed for {trade.symbol} on {date}: {e}")
        return None

def calculate_reward(entry, exit, trade):
    pnl = (exit - entry) * trade.direction
    risk_penalty = 0.02 * trade.max_drawdown if hasattr(trade, "max_drawdown") else 0
    cost_penalty = 0.003 + 0.001 * abs(trade.size)
    novelty_score = trade.meta.get("novelty", 0.5)
    info_bonus = 0.01 * novelty_score
    return pnl - risk_penalty - cost_penalty + info_bonus
