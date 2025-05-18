from integrations.zerodha_client import get_kite
import pandas as pd

kite = get_kite()
instruments = kite.instruments("NSE")
df = pd.DataFrame(instruments)
df.to_csv("data/instruments.csv", index=False)
print("✅ instruments.csv saved")
