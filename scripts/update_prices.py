from core.price_service import get_prices
from core.data_provider import cache_price
from db.postgres_manager import get_all_symbols
from core.time_context import get_simulation_date

today = get_simulation_date()
for symbol in get_all_symbols():
    df = get_prices(symbol, end=today, interval="day")
    cache_price(df)
