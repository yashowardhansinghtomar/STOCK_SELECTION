# utils/stock_precheck.py

from core.feature_generator import generate_features
from core.model_io import load_model
from config.paths import PATHS
from core.logger import logger

# Load filter model feature names once
try:
    _, REQUIRED_FEATURES = load_model("filter_model")
except Exception as e:
    logger.warning(f"⚠️ Failed to load filter model for precheck: {e}")
    REQUIRED_FEATURES = []

def is_feature_ready(stock: str, verbose=False) -> bool:
    try:
        df = generate_features(stock)
        if df.empty:
            if verbose:
                logger.debug(f"⛔ {stock} → Empty features")
            return False
        missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
        if missing:
            if verbose:
                logger.debug(f"⛔ {stock} → Missing features: {missing}")
            return False
        return True
    except Exception as e:
        if verbose:
            logger.debug(f"⛔ {stock} → Feature error: {e}")
        return False

def filter_valid_stocks(stocks: list[str], verbose=False) -> list[str]:
    valid, skipped = [], []
    for stock in stocks:
        if is_feature_ready(stock, verbose):
            valid.append(stock)
        else:
            skipped.append(stock)
    logger.info(f"✅ Precheck complete: {len(valid)} valid, {len(skipped)} skipped.")
    return valid
