from core.logger.logger import logger
from core.data_provider.data_provider import load_data, save_data
from core.time_context.time_context import get_simulation_date
from core.config.config import settings
import pandas as pd
from datetime import datetime

def run_technical_filter(threshold=10, return_reasons=False, min_history=30):
    all_features = load_data(settings.feature_table)
    if all_features is None or all_features.empty:
        logger.warning("⚠️ No technical features found.")
        return (pd.DataFrame(), {}) if return_reasons else pd.DataFrame()

    sim_date = pd.to_datetime(get_simulation_date())
    latest_date = pd.to_datetime(all_features["date"]).max()
    effective_date = min(sim_date, latest_date).strftime("%Y-%m-%d")
    df = all_features[all_features["date"] == effective_date].copy()

    if df.empty:
        logger.warning(f"⚠️ No feature data available for {effective_date}")
        return (pd.DataFrame(), {}) if return_reasons else pd.DataFrame()

    # Only keep stocks with ≥ min_history rows
    hist_counts = all_features.groupby("stock")["date"].nunique()
    sufficient_data_stocks = hist_counts[hist_counts >= min_history].index.tolist()
    df = df[df["stock"].isin(sufficient_data_stocks)]

    if df.empty:
        logger.warning("⚠️ All stocks dropped due to insufficient price history.")
        return (df, {}) if return_reasons else df

    ε = 1e-5  # tiny float tolerance

    cond1 = df["sma_short"] >= df["sma_long"] - ε
    cond2 = df["rsi_thresh"] < 80
    cond3 = df["sma_short"] < df["sma_long"] * 1.05

    df["reason"] = ""
    for idx, row in df.iterrows():
        reasons = []
        if not cond1.loc[idx]:
            reasons.append(f"❌ SMA short < SMA long ({row['sma_short']:.2f} < {row['sma_long']:.2f})")
        if not cond2.loc[idx]:
            reasons.append(f"❌ RSI too high ({row['rsi_thresh']:.2f} > 70.00)")
        if not cond3.loc[idx]:
            reasons.append(f"❌ SMA short > 105% of SMA long ({row['sma_short']:.2f} > {row['sma_long']*1.05:.2f})")
        df.at[idx, "reason"] = " | ".join(reasons)

    rejected = df[~(cond1 & cond2 & cond3)]
    rejected_dict = {
        str(row["stock"]): str(row["reason"])
        for _, row in rejected.iterrows()
    }

    if not rejected.empty:
        logger.warning("⚠️ Sample rejected stocks with reasons:")
        for stock, reason in list(rejected_dict.items())[:10]:
            logger.warning(f"{stock}: {reason}")

    df = df[cond1 & cond2 & cond3]
    if df.empty:
        logger.warning("⚠️ No stocks passed the fallback technical filter.")
        return (df, rejected_dict) if return_reasons else df

    result = pd.DataFrame({
        "stock": df["stock"],
        "source": "fallback_technical",
        "imported_at": datetime.now()
    })
    save_data(result, settings.ml_selected_stocks_table, if_exists="replace")
    logger.success(f"✅ ✅ Fallback technical filter saved {len(result)} stocks.")

    return (df, rejected_dict) if return_reasons else df
