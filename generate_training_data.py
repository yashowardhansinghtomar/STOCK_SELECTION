from core.logger import logger
# generate_training_data.py
import pandas as pd
import os

RECOMMENDATIONS_CSV = "results/weekly_recommendations.csv"
FUNDAMENTALS_CSV = "results/stock_fundamentals.csv"
OUTPUT_PATH = "training_data.csv"

def main():
    if not os.path.exists(RECOMMENDATIONS_CSV) or not os.path.exists(FUNDAMENTALS_CSV):
        logger.error("❌ Missing input files.")
        return

    recs = pd.read_csv(RECOMMENDATIONS_CSV)
    funds = pd.read_csv(FUNDAMENTALS_CSV)

    # Normalize column names
    recs.columns = [col.lower().strip() for col in recs.columns]
    funds.columns = [col.lower().strip() for col in funds.columns]

    # Label as 1 if return > 10% and sharpe > 0.5 and drawdown < 30%
    recs["label"] = recs.apply(
    lambda row: 1 if row["total_return"] > 10 and row["sharpe"] > 0.5 and row["max_drawdown"] < 30 else 0,
    axis=1
)

    # Merge fundamentals into recommendations
    merged = recs.merge(funds, on="stock", how="inner")

    # Select relevant features
    feature_cols = [
        "pe_ratio", "de_ratio", "roe",
        "earnings_growth", "market_cap",
        "label"
    ]

    merged = merged[feature_cols].dropna()
    merged.to_csv(OUTPUT_PATH, index=False)

    logger.success(f"✅ Training data saved to {OUTPUT_PATH} ({len(merged)} rows)")

if __name__ == "__main__":
    main()
