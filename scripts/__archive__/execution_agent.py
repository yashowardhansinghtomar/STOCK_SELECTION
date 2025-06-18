from core.data_provider.data_provider import load_data, save_data
from core.logger.logger import logger
# agents/execution_agent.py
import os
import time
import pandas as pd
import yfinance as yf
from datetime import datetime
from config.paths import PATHS
from utils.file_io import load_dataframe, save_dataframe
from core.data_provider.data_provider import load_data
from core.logger.logger import logger


CAPITAL_PER_TRADE = 10000

class ExecutionAgent:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.dev_mode = os.getenv("DEV_MODE") == "1"

    def load_recommendations(self):
        df = load_data("recommendations")
        return df.head(3) if self.dev_mode else df

    def load_open_positions(self):
        required_cols = ["stock", "entry_date", "entry_price", "sma_short", "sma_long", "rsi_thresh"]
        df = load_data("open_positions")  # SQLIFIED
        if not all(col in df.columns for col in required_cols):
            logger.warning("⚠️ open_positions missing expected columns. Reinitializing.")
            return pd.DataFrame(columns=required_cols)
        return df

    def load_today_price(self, ticker):
        df = yf.download(f"{ticker}.NS", period="2d", interval="1d")
        if df.empty:
            return None
        return float(df.iloc[-1]["close"])

    def check_exit_condition(self, ticker):
        df = yf.download(f"{ticker}.NS", period="60d")
        if df.empty or len(df) < 30:
            return False, None
        df["SMA30"] = df["close"].rolling(window=30).mean()
        last_close = df.iloc[-1]["close"]
        last_sma = df.iloc[-1]["SMA30"]
        if pd.isna(last_sma):
            return False, None
        return (last_close < last_sma), last_close

    def log_trades(self, trades):
        if not trades:
            return
        log_df = load_dataframe(PATHS["paper_trades"])
        df = pd.concat([log_df, pd.DataFrame(trades)], ignore_index=True)
        save_dataframe(df, PATHS["paper_trades"])

    def enter_trades(self, recommendations, open_positions):
        new_positions = []
        for _, row in recommendations.iterrows():
            if row.get("trade_triggered", 1) != 1:
                continue
            if row["stock"] in open_positions["stock"].values:
                continue
            price = self.load_today_price(row["stock"])
            if price is None:
                continue
            logger.success(f"✅ Entering: {row['stock']} at {price:.2f}")
            new_positions.append({
                "stock": row["stock"],
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "entry_price": price,
                "sma_short": row["sma_short"],
                "sma_long": row["sma_long"],
                "rsi_thresh": row["rsi_thresh"]
            })
        return pd.concat([open_positions, pd.DataFrame(new_positions)], ignore_index=True)

    def exit_trades(self, open_positions):
        remaining, exited = [], []
        for _, row in open_positions.iterrows():
            should_exit, exit_price = self.check_exit_condition(row["stock"])
            if should_exit:
                logger.info(f"🚮 Exiting: {row['stock']} at {exit_price:.2f}")
                exited.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "stock": row["stock"],
                    "action": "sell",
                    "price": exit_price,
                    "signal_reason": "close < SMA30",
                    "strategy_config": f"{row['sma_short']}/{row['sma_long']}/RSI{row['rsi_thresh']}"
                })
            else:
                remaining.append(row)
        return pd.DataFrame(remaining), exited

    def run(self):
        logger.start("\n🚀 Running ExecutionAgent (Paper Mode)...")
        start = time.time()
        recommendations = self.load_recommendations()
        open_positions = self.load_open_positions()
        open_positions, exited = self.exit_trades(open_positions)
        self.log_trades(exited)
        open_positions = self.enter_trades(recommendations, open_positions)
        save_data(open_positions, "open_positions")  # SQLIFIED
        logger.success(f"✅ ExecutionAgent complete. ⏱️ {time.time() - start:.2f}s")

if __name__ == "__main__":
    ExecutionAgent().run()
