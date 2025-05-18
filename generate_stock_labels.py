from core.logger import logger
# generate_stock_labels.py
import os
import pandas as pd

TRADES_FILE = "results/paper_trades.csv"
LABELS_FILE = "results/stock_labels.csv"


def generate_labels():
    if not os.path.exists(TRADES_FILE):
        logger.error("❌ paper_trades.csv not found.")
        return

    df = pd.read_csv(TRADES_FILE)
    if df.empty or "stock" not in df:
        logger.warning("⚠️ No data in trade logs.")
        return

    df = df[df["action"] == "sell"].copy()
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["entry_price", "exit_price"])

    df["return"] = (df["exit_price"] - df["entry_price"]) / df["entry_price"]
    df["label"] = df["return"].apply(lambda r: 1 if r > 0.02 else 0)

    summary = df.groupby("stock")["label"].mean().reset_index()
    summary["label"] = summary["label"].apply(lambda x: 1 if x >= 0.5 else 0)

    summary.to_csv(LABELS_FILE, index=False)
    logger.success(f"✅ Stock trade outcome labels saved to {LABELS_FILE}")


if __name__ == "__main__":
    generate_labels()
