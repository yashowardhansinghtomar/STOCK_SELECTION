# analysis/joint_policy_comparator.py

import pandas as pd
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

def compare_joint_policy_vs_rf():
    date = pd.to_datetime(get_simulation_date()).date()

    rf_df = load_data("trades")
    joint_df = load_data("joint_policy_predictions")

    if rf_df is None or joint_df is None:
        logger.warning("⚠️ Missing trades or predictions.")
        return

    rf_today = rf_df[pd.to_datetime(rf_df["timestamp"]).dt.date == date]
    rf_today = rf_today[["stock"]].copy()
    rf_today["rf_traded"] = 1

    joint_today = joint_df[joint_df["date"] == date][["stock", "enter_prob"]]
    joint_today["model_traded"] = (joint_today["enter_prob"] > 0.5).astype(int)

    merged = pd.merge(joint_today, rf_today, on="stock", how="outer").fillna(0)
    merged["rf_traded"] = merged["rf_traded"].astype(int)

    def classify(row):
        if row["rf_traded"] == 1 and row["model_traded"] == 1:
            return "true_positive"
        elif row["rf_traded"] == 1 and row["model_traded"] == 0:
            return "false_negative"
        elif row["rf_traded"] == 0 and row["model_traded"] == 1:
            return "false_positive"
        else:
            return "true_negative"

    merged["decision_type"] = merged.apply(classify, axis=1)

    counts = merged["decision_type"].value_counts()
    logger.info(f"📊 Joint policy vs RF on {date}:\n{counts.to_string()}")

    return merged

if __name__ == "__main__":
    compare_joint_policy_vs_rf()
