

from core.feature_engineering.feature_provider import fetch_features
from redis_worker.redis_utils import enqueue_feature_backfill, wait_for_feature_ready

def fetch_features_with_backfill(symbol: str, interval: str, timeout: int = 300):
    """
    Wrapper that ensures features are computed via Redis before fetching them.
    """
    enqueue_feature_backfill(symbol, interval)
    ready = wait_for_feature_ready(symbol, interval, timeout=timeout)
    if not ready:
        raise TimeoutError(f"Features not available for {symbol} @ {interval} after {timeout} seconds")
    return fetch_features(symbol, interval)
