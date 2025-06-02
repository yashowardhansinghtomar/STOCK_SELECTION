# training/train_joint_policy.py

import pandas as pd
from core.data_provider.data_provider import load_data
from core.feature_engineering.feature_provider import fetch_features
from core.logger.logger import logger
from models.joint_policy import JointPolicyModel
from core.config.config import settings
from datetime import timedelta

def load_training_data():
    trades_df = load_data(settings.trades_table)
    if trades_df is None or trades_df.empty:
        logger.error("❌ No trades found.")
        return pd.DataFrame()

    trades_df["date"] = pd.to_datetime(trades_df["timestamp"]).dt.date
    trades_df["enter"] = 1
    return trades_df[["stock", "date", "enter", "price"]]

def sample_negative_examples(trades_df, sample_size=2):
    recs_df = load_data(settings.recommendations_table)
    if recs_df is None or recs_df.empty:
        return pd.DataFrame()

    recs_df["date"] = pd.to_datetime(recs_df["date"]).dt.date
    recs_df = recs_df[recs_df["trade_triggered"] == 1]
    recs_df = recs_df[~recs_df["stock"].isin(trades_df["stock"])]

    sampled = recs_df.groupby("date").apply(lambda x: x.sample(min(sample_size, len(x)))).reset_index(drop=True)
    sampled["enter"] = 0
    sampled["price"] = 0.0  # not used
    return sampled[["stock", "date", "enter", "price"]]

def fetch_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        try:
            features = fetch_features(row["stock"], interval="day", refresh_if_missing=False)
            features["date"] = pd.to_datetime(features["date"]).dt.date
            matched = features[features["date"] == row["date"]]
            if matched.empty:
                continue

            feat_row = matched.iloc[-1].to_dict()
            feat_row.update({
                "stock": row["stock"],
                "date": row["date"],
                "enter": row["enter"],
                "position_size": 1.0 if row["enter"] == 1 else 0.0,
                "exit_days": 3  # Default holding period
            })
            rows.append(feat_row)
        except Exception as e:
            logger.warning(f"⚠️ Feature fetch failed for {row['stock']} on {row['date']}: {e}")
    return pd.DataFrame(rows)

def train_model(train_df: pd.DataFrame):
    features = train_df.drop(columns=["stock", "date", "enter", "position_size", "exit_days"])
    labels = train_df[["enter", "position_size", "exit_days"]]

    model = JointPolicyModel()
    model.fit(features, labels)
    model.save()
    logger.success("✅ Joint policy model trained and saved.")

def main():
    logger.start("🚀 Starting joint policy training...")
    trades = load_training_data()
    if trades.empty:
        return

    neg_samples = sample_negative_examples(trades)
    all_data = pd.concat([trades, neg_samples], ignore_index=True)

    train_df = fetch_feature_data(all_data)
    if train_df.empty:
        logger.error("❌ No feature rows fetched.")
        return

    train_model(train_df)

if __name__ == "__main__":
    main()
