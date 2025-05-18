import pandas as pd
from core.data_provider import load_data
from core.time_context import get_simulation_date
from core.config import settings

sim_date = pd.to_datetime(get_simulation_date()).date()
feature_date = (pd.Timestamp(sim_date) - pd.tseries.offsets.BDay(1)).date()

feats = load_data(settings.feature_table)
print("✅ stock_features loaded:", not feats.empty)
feats["date"] = pd.to_datetime(feats["date"]).dt.date

try:
    trades = load_data(settings.paper_trades_table)
    if trades is None or trades.empty:
        print("❌ No trades found")
    else:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades = trades[trades["timestamp"].dt.date == sim_date].copy()
        trades["date"] = (trades["timestamp"].dt.normalize() - pd.tseries.offsets.BDay(1)).dt.date

        print("💡 Feature stocks on", feature_date, ":", feats[feats["date"] == feature_date]["stock"].unique())
        print("🧾 Trade stocks for", sim_date, ":", trades["stock"].unique())
except Exception as e:
    print("⚠️ Load error:", e)
