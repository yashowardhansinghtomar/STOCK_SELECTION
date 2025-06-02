import os
import pandas as pd
import yfinance as yf
from datetime import datetime
from core.logger.logger import logger

# ==== Config ====
RECOMMENDATIONS_CSV = "results/weekly_recommendations.csv"
OPEN_POSITIONS_CSV = "results/open_positions.csv"
TRADE_LOG_CSV = "results/paper_trades.csv"
CAPITAL_PER_TRADE = 10000  # Simulate buying with fixed amount

# ==== Load data ====
def load_recommendations():
    if os.path.exists(RECOMMENDATIONS_CSV):
        return pd.read_csv(RECOMMENDATIONS_CSV)
    return pd.DataFrame()

def load_open_positions():
    if os.path.exists(OPEN_POSITIONS_CSV):
        return pd.read_csv(OPEN_POSITIONS_CSV)
    return pd.DataFrame(columns=["stock", "entry_date", "entry_price", "sma_short", "sma_long", "rsi_thresh"])

def load_today_price(ticker):
    df = yf.download(f"{ticker}.NS", period="2d", interval="1d")
    if df.empty:
        return None
    return df.iloc[-1]["close"]

# ==== Entry Logic ====
def enter_trades(recommendations, open_positions):
    new_positions = []
    for _, row in recommendations.iterrows():
        if row["trade_triggered"] != 1:
            continue
        if row["stock"] in open_positions["stock"].values:
            continue

        price = load_today_price(row["stock"])
        if price is None:
            continue

        logger.info(f"📥 Buying {row['stock']} at {price:.2f}")
        new_positions.append({
            "stock": row["stock"],
            "entry_date": datetime.now().strftime("%Y-%m-%d"),
            "entry_price": price,
            "sma_short": row["sma_short"],
            "sma_long": row["sma_long"],
            "rsi": row["rsi"]
        })

    return pd.concat([open_positions, pd.DataFrame(new_positions)], ignore_index=True)

# ==== Exit Logic (simple SMA based exit for now) ====
def check_exit_condition(ticker, entry_price):
    df = yf.download(f"{ticker}.NS", period="60d")
    if df.empty or len(df) < 30:
        return False, None

    df["SMA30"] = df["close"].rolling(window=30).mean()
    last_close = df.iloc[-1]["close"]
    last_sma = df.iloc[-1]["SMA30"]

    if pd.isna(last_sma):
        return False, None

    # Example: exit if close drops below SMA
    if last_close < last_sma:
        return True, last_close

    return False, last_close

def exit_trades(open_positions):
    remaining = []
    exited_trades = []

    for _, row in open_positions.iterrows():
        should_exit, exit_price = check_exit_condition(row["stock"], row["entry_price"])
        if should_exit:
            logger.info(f"💼 Selling {row['stock']} at {exit_price:.2f}")
            exited_trades.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "stock": row["stock"],
                "action": "sell",
                "price": exit_price,
                "signal_reason": "close below SMA30",
                "strategy_config": f"{row['sma_short']}/{row['sma_long']}/RSI{row['rsi_thresh']}"
            })
        else:
            remaining.append(row)

    return pd.DataFrame(remaining), exited_trades

# ==== Logger ====
def log_trades(trades):
    if not trades:
        return
    df = pd.DataFrame(trades)
    if os.path.exists(TRADE_LOG_CSV):
        existing = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(TRADE_LOG_CSV, index=False)

# ==== Main Runner ====
def main():
    logger.start("\n🚀 Starting paper trader...")
    recommendations = load_recommendations()
    open_positions = load_open_positions()

    # Exit trades
    open_positions, exited = exit_trades(open_positions)
    log_trades(exited)

    # Enter new trades
    open_positions = enter_trades(recommendations, open_positions)
    open_positions.to_csv(OPEN_POSITIONS_CSV, index=False)
    logger.success("✅ Paper trading session complete.")

if __name__ == "__main__":
    main()
