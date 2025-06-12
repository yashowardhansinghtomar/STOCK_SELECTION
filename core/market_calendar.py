# core/market_calender.py

from datetime import timedelta, datetime

def get_trading_days(start, end):
    """
    Returns all weekdays (Mon-Fri) between start and end dates, inclusive.
    """
    current = start
    days = []
    while current <= end:
        if current.weekday() < 5:  # Monday=0, Sunday=6
            days.append(current)
        current += timedelta(days=1)
    return days

# Partial known NSE holidays in 2023. Add only those needed for your backtest period.
NSE_HOLIDAYS = {
    "2023-01-26", "2023-03-07", "2023-03-30",  # official
    "2023-01-02",  # <== manually added as non-trading
}

def is_market_holiday(date: datetime) -> bool:
    return date.weekday() >= 5 or date.strftime("%Y-%m-%d") in NSE_HOLIDAYS