from datetime import timedelta

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
