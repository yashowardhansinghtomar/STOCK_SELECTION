import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from core.logger.logger import logger
from core.config.config import settings, get_feature_columns
from core.data_provider.data_provider import load_data
from core.model_io import save_model
from collections import Counter

def train_param_model():
    logger.start("🧠 Training Param Model(s) — interval-specific...")

    df = load_data(settings.training_data_table)
    if df is None or df.empty:
        logger.error("❌ No training data found.")
        return

    df = df.dropna(subset=["target", "interval", "sma_short", "sma_long", "rsi_thresh"])
    df["interval"] = df["interval"].fillna("day")

    le = LabelEncoder()
    df["interval_encoded"] = le.fit_transform(df["interval"])
    intervals = df["interval"].dropna().unique()

    for interval in intervals:
        sub_df = df[df["interval"] == interval]
        if sub_df.empty:
            continue

        logger.info(f"📊 Training param model for interval: {interval} (rows: {len(sub_df)})")
        logger.info(f"📈 Output dist — SMA_Short: {Counter(sub_df['sma_short'])}")

        features = get_feature_columns(interval)
        missing_feats = [f for f in features if f not in sub_df.columns]
        if missing_feats:
            logger.warnings(f"⚠️ Skipping {interval} — missing features: {missing_feats}")
            continue

        X = sub_df[features].copy().fillna(0).replace([float("inf"), float("-inf")], 0)
        y = sub_df[["interval_encoded", "sma_short", "sma_long", "rsi_thresh"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.test_size, random_state=settings.random_state
        )

        model = MultiOutputRegressor(RandomForestRegressor(
            n_estimators=150, max_depth=6, random_state=42
        ))
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds)

        model_name = f"{settings.model_names['param']}_{interval}"
        save_model(model_name, {
            "model": model,
            "features": features,
            "label_encoder": le,
            "interval": interval,
            "params": {"n_estimators": 150, "max_depth": 6},
            "rmse": rmse
        }, meta={
            "trained_at": str(pd.Timestamp.now()),
            "interval": interval,
            "rmse": round(rmse, 4),
            "rows": len(sub_df),
            "feature_count": len(features),
            "std_sma_short": sub_df["sma_short"].std(),
            "std_rsi": sub_df["rsi_thresh"].std()
        })

        logger.success(f"✅ Trained {model_name} (RMSE: {rmse:.4f})")

    logger.success("📦 All interval-specific param models saved.")


if __name__ == "__main__":
    train_param_model()
