from core.data_provider.data_provider import load_data
from core.logger.logger import logger

FEATURE_TABLE = "stock_features_day"

feats = load_data(FEATURE_TABLE)
expected_cols = {"sma_short", "sma_long", "rsi_thresh"}
actual_cols = set(feats.columns)

logger.info(f"✅ Found columns: {actual_cols}")
missing = expected_cols - actual_cols
if missing:
    logger.error(f"❌ Missing required columns: {missing}")
else:
    logger.success("✅ All required columns are present.")
