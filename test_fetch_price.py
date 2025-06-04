from core.data_provider.data_provider import get_last_close
from datetime import datetime

for stock in ['RELIANCE', 'ICICIBANK', 'SBIN', 'INFY', 'LT']:
    price = get_last_close(stock, sim_date=datetime(2023, 1, 2))
    print(f"{stock}: {price}")
