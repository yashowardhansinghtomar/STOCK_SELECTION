# core/broker_api.py

REAL_ORDER_MODE = False  # Toggle this to True to enable real order submission

def submit_order_live(trade):
    """
    Submits a live order if REAL_ORDER_MODE is enabled.
    Otherwise logs the order and returns a simulated response.
    """
    if REAL_ORDER_MODE:
        # Implement integration here with actual broker API (e.g., Zerodha Kite)
        raise NotImplementedError("Live order submission not implemented yet.")
    else:
        print(f"[PAPER TRADE] Simulating live order: {trade.symbol} {trade.direction} @ {trade.price}")
        return {
            "status": "paper",
            "symbol": trade.symbol,
            "price": trade.price,
            "direction": trade.direction,
            "size": trade.size
        }