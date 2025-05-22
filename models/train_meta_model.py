# models/train_meta_model.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from core.logger import logger
from core.model_io import save_model
from core.data_provider import load_data, save_data
from core.feature_enricher_multi import enrich_multi_interval_features
from core.config import settings, get_feature_columns


def train_meta_model():
    logger.start("🧠 Training Meta Model with Multi-Interval Features...")

    df_base = load_data(settings.meta_training_table)
    if df_base is None or df_base.empty:
        logger.error("❌ No meta training data found.")
        return

    rows = []
    for _, row in df_base.iterrows():
        stock = row["stock"]
        date = pd.to_datetime(row["date"])
        enriched = enrich_multi_interval_features(stock, date)
        if enriched.empty:
            continue
        enriched["target"] = row["target"]
        rows.append(enriched)

    if not rows:
        logger.error("❌ No usable rows after enrichment.")
        return

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["target"])

    X = df[get_feature_columns()]  # unified feature list
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state
    )

    model = RandomForestRegressor(
        n_estimators=settings.meta_n_estimators,
        random_state=settings.random_state,
        n_jobs=settings.meta_n_jobs
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    logger.success(f"✅ Meta model trained. RMSE: {rmse:.4f}")

    save_model("meta_model", {
        "model": model,
        "features": list(X.columns),
        "params": {
            "n_estimators": settings.meta_n_estimators,
            "max_depth": settings.meta_max_depth,
        }
    })

    save_data(pd.DataFrame([{
        "model_name": "meta_model",
        "date": pd.Timestamp.now(),
        "rmse": rmse,
        "accuracy": None
    }]), settings.meta_metadata_table)

    logger.success("📦 Meta model saved with metadata.")


if __name__ == "__main__":
    train_meta_model()
