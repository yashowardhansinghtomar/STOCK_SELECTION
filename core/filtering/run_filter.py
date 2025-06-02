# core/filtering/run_filter.py

from datetime import datetime
import pandas as pd
from models.run_stock_filter import run_stock_filter
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

PREDICTION_TABLE = "filter_model_predictions"

def run_filter_model(date: datetime, lookback_only: bool = False) -> list:
    """
    Get filtered stocks for a given date using trained filter model.

    Args:
        date (datetime): The target date to run filter for.
        lookback_only (bool): Ignored for now, for compatibility.

    Returns:
        list: List of stock symbols predicted as "buy".
    """
    date_str = date.strftime("%Y-%m-%d")

    try:
        # Step 1: Run the filter model to generate predictions
        run_stock_filter(as_of=date)

        # Step 2: Load predictions from SQL
        preds = load_data(PREDICTION_TABLE)
        if preds is None or preds.empty:
            logger.warning(f"No filter predictions found in table '{PREDICTION_TABLE}' for {date_str}")
            return []

        preds = preds[preds["date"] == date_str]
        preds = preds[preds["decision"] == "buy"]

        if preds.empty:
            logger.warning(f"No buy recommendations on {date_str}")
            return []

        # Step 3: Return sorted list of top stocks
        top_stocks = preds.sort_values("confidence", ascending=False)["stock"].tolist()
        logger.info(f"Filter model selected {len(top_stocks)} stocks for {date_str}")
        return top_stocks

    except Exception as e:
        logger.error(f"Failed to run filter model for {date_str}: {e}")
        return []
