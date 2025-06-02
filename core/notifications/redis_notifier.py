from redis import Redis

r = Redis()

def push_feature_ready(symbol: str, queue: str = "feature_ready_1m"):
    """
    Push a symbol to the Redis queue to signal that features are ready.
    """
    r.rpush(queue, symbol)
