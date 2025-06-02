# bootstrap/simulate_trade_execution.py

from core.data_provider.data_provider import fetch_stock_data
from core.realism_boosters.slippage import calculate_slippage
from core.realism_boosters.market_impact import estimate_market_impact
from datetime import timedelta
import random


def simulate_trade_execution(trade, date):
    # Load minute bars for execution date
    bars_df = fetch_stock_data(trade.symbol, interval="minute", end=date, days=1)
    if bars_df is None or bars_df.empty:
        return None

    try:
        if trade.order_type == "MARKET":
            exe_time = trade.timestamp + timedelta(minutes=random.randint(1, 5))
            row = bars_df[bars_df["timestamp"] == exe_time]
            if row.empty:
                return None
            exe_price = row.iloc[0]["open"]

            # Apply slippage and impact
            slippage = calculate_slippage(trade.symbol, trade.size, date)
            impact = estimate_market_impact(trade.size, trade.symbol, date)
            exe_price *= (1 + slippage + impact)

        elif trade.order_type == "LIMIT":
            fill_prob = trade.meta.get("fill_prob", 0.7)
            if random.random() < fill_prob:
                exe_price = trade.price
            else:
                return None

        # Handle exit strategy
        if trade.exit_strategy == "TIME_BASED":
            exit_date = date + trade.holding_period
            exit_bars_df = fetch_stock_data(trade.symbol, interval="minute", end=exit_date, days=1)
            if exit_bars_df is None or exit_bars_df.empty:
                return None

            exit_row = exit_bars_df[exit_bars_df["timestamp"] == trade.exit_time]
            if exit_row.empty:
                exit_price = exe_price  # fallback
            else:
                exit_price = exit_row.iloc[0]["close"]

        reward = calculate_reward(entry=exe_price, exit=exit_price, trade=trade)

        return {
            "symbol": trade.symbol,
            "entry_price": exe_price,
            "exit_price": exit_price,
            "entry_time": exe_time,
            "exit_time": trade.exit_time,
            "reward": reward,
            "direction": trade.direction,
            "meta": trade.meta
        }

    except Exception as e:
        print(f"Simulation failed for {trade.symbol} on {date}: {e}")
        return None


def calculate_reward(entry, exit, trade):
    pnl = (exit - entry) * trade.direction
    risk_penalty = 0.02 * trade.max_drawdown if hasattr(trade, "max_drawdown") else 0
    cost_penalty = 0.003 + 0.001 * abs(trade.size)
    novelty_score = trade.meta.get("novelty", 0.5)
    info_bonus = 0.01 * novelty_score
    return pnl - risk_penalty - cost_penalty + info_bonus
