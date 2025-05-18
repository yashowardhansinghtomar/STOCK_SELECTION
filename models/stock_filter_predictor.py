# stock_filter_predictor.py

from core.logger import logger
from core.data_provider import load_data, save_data
from core.model_io import load_model
from core.config import settings
import pandas as pd

def run_stock_filter():
    # Load fundamentals (now contains proxy columns)
    fundamentals = load_data(settings.fundamentals_table)
    if fundamentals is None or fundamentals.empty:
        logger.error(f"❌ {settings.fundamentals_table} not found or empty.")
        return

    # Load ML filter model
    try:
        model, feature_cols = load_model(settings.filter_model_name)
    except Exception as e:
        logger.warning(f"⚠️ Could not load ML model '{settings.filter_model_name}': {e}")
        model, feature_cols = None, None

    # If no model, pass everything through
    if model is None or not feature_cols:
        logger.warning("⚠️ No ML model available — passing all stocks through filter.")
        selected = fundamentals["stock"]
    else:
        # Verify required features exist
        missing = [c for c in feature_cols if c not in fundamentals.columns]
        if missing:
            logger.error(f"❌ Missing features in {settings.fundamentals_table}: {missing}")
            return

        X = fundamentals[feature_cols].fillna(0)
        preds = model.predict(X)
        selected = fundamentals.loc[preds == 1, "stock"]

    # Save the selected list into the configured ML-selected table
    df_selected = pd.DataFrame({"stock": selected})
    save_data(df_selected, settings.ml_selected_stocks_table)
    logger.success(
        f"✅ {len(df_selected)} stocks passed ML filter → "
        f"saved to '{settings.ml_selected_stocks_table}'."
    )

if __name__ == "__main__":
    run_stock_filter()
