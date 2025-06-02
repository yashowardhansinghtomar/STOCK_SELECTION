# core/realism_boosters/market_impact.py

def estimate_market_impact(size, symbol, date):
    """
    Returns estimated market impact as a fraction of price (e.g., 0.002 means 0.2%)
    Placeholder logic: based on trade size.
    """
    if size > 1.0:
        return 0.005
    elif size > 0.5:
        return 0.002
    else:
        return 0.001
