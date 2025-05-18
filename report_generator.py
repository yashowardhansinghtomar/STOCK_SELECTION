from core.logger import logger
# report_generator.py
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

TRADES_FILE = "results/paper_trades.csv"
OPEN_POSITIONS_FILE = "results/open_positions.csv"
PLOT_DIR = "results"


def load_data():
    trades_df = pd.read_csv(TRADES_FILE) if os.path.exists(TRADES_FILE) else pd.DataFrame()
    open_df = pd.read_csv(OPEN_POSITIONS_FILE) if os.path.exists(OPEN_POSITIONS_FILE) else pd.DataFrame()
    return trades_df, open_df


def analyze_trades(trades_df):
    if trades_df.empty:
        logger.info("\n📭 No closed trades to analyze yet.")
        return

    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], errors="coerce")
    trades_df["date"] = trades_df["timestamp"].dt.date

    logger.info("\n📊 Trade Summary:")
    logger.info(f"Total trades: {len(trades_df)}")

    trades_df["action"] = trades_df["action"].str.lower()
    sell_trades = trades_df[trades_df["action"] == "sell"]

    if not sell_trades.empty:
        logger.info("\n🔍 Sell Trades:")
        avg_price = sell_trades["price"].mean()
        logger.info(f"Average Sell Price: ₹{avg_price:.2f}")

        grouped = sell_trades.groupby("stock")["price"].agg(["count", "mean", "min", "max"])
        logger.info(grouped)

        # Optional plot: sell price per stock
        plt.figure(figsize=(10, 5))
        sell_trades.groupby("stock")["price"].mean().sort_values().plot(kind="barh", color="skyblue")
        plt.title("Average Sell Price per Stock")
        plt.xlabel("Price")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "sell_price_chart.png"))
        plt.close()

    logger.info("\n🧾 Sample Trades:")
    logger.info(trades_df.tail(5))


def fetch_current_price(ticker):
    try:
        df = yf.download(f"{ticker}.NS", period="2d", interval="1d", progress=False)
        return float(df["close"].iloc[-1]) if not df.empty else None
    except:
        return None


def analyze_open_positions(open_df):
    if open_df.empty:
        logger.info("\n📭 No open positions.")
        return

    logger.info("\n📌 Current Open Positions with PnL:")
    open_df["entry_date"] = pd.to_datetime(open_df["entry_date"], errors="coerce")
    open_df["days_open"] = (datetime.now() - open_df["entry_date"]).dt.days

    pnl_data = []
    for _, row in open_df.iterrows():
        current_price = fetch_current_price(row["stock"])
        if current_price is None:
            continue
        pnl = (current_price - row["entry_price"]) / row["entry_price"]
        pnl_data.append({
            "stock": row["stock"],
            "entry_price": row["entry_price"],
            "current_price": current_price,
            "return_%": round(pnl * 100, 2),
            "days_open": row["days_open"]
        })

    pnl_df = pd.DataFrame(pnl_data)
    logger.info(pnl_df)
    avg_return = pnl_df["return_%"].mean()
    logger.info(f"\n📈 Avg Floating Return: {avg_return:.2f}%")

    if avg_return > 5:
        logger.info("💡 Strategy Insight: Positions are performing well. Consider increasing size.")
    elif avg_return < -5:
        logger.warning("⚠️ Strategy Insight: Negative return. Consider adjusting filters or parameters.")
    else:
        logger.info("🧠 Strategy Insight: Neutral zone. Monitor closely.")

    # 📊 Bar chart: return % per stock
    if not pnl_df.empty:
        plt.figure(figsize=(10, 5))
        colors = ["green" if val > 0 else "red" for val in pnl_df["return_%"]]
        plt.bar(pnl_df["stock"], pnl_df["return_%"], color=colors)
        plt.title("Floating Returns on Open Positions")
        plt.ylabel("Return %")
        plt.axhline(0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "open_positions_returns.png"))
        plt.close()


def main():
    logger.info("\n📋 Running Weekly Performance Report...")
    trades_df, open_df = load_data()
    analyze_trades(trades_df)
    analyze_open_positions(open_df)
    logger.success("\n✅ Report generated. Charts saved to 'results/' folder.")


if __name__ == "__main__":
    main()
