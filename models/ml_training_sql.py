# models/ml_training_sql.py
from core.model_io import save_model
from core.logger.logger import logger
from core.data_provider.data_provider import load_data, save_data
from core.config.config import settings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd

# Add at top
INTERVALS = ["day", "60minute", "15minute"]
SUFFIXES = ["_day", "_60m", "_15m"]

def merge_intervals(df_list):
    merged = df_list[0]
    for df in df_list[1:]:
        merged = pd.merge(merged, df, on=["stock", "date"], how="outer")
    return merged

def train_meta_model():
    """
    Train a meta-model (regressor) on combined multi-interval features.
    Uses settings for split and hyperparams.
    """
    logger.start("🚀 Starting Meta Model Training...")

    # Load meta training base
    df_base = load_data(settings.meta_training_table)
    if df_base is None or df_base.empty:
        logger.error("❌ No meta training base data found. Aborting.")
        return

    # Load and merge multi-interval features
    from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features

    merged_rows = []
    for _, row in df_base.iterrows():
        enriched = enrich_multi_interval_features(row["stock"], pd.to_datetime(row["date"]))
        if not enriched.empty:
            enriched["target"] = row["target"]
            merged_rows.append(enriched)

    if not merged_rows:
        logger.error("❌ No enriched rows available after merging.")
        return

    df = pd.concat(merged_rows, ignore_index=True)

    # Encode stock label
    if "stock" in df.columns:
        le = LabelEncoder()
        df["stock_encoded"] = le.fit_transform(df["stock"].astype(str))

    X = df.drop(columns=[col for col in ["target", "date", "stock"] if col in df.columns])
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state
    )

    model = RandomForestRegressor(
        n_estimators=settings.meta_n_estimators,
        random_state=settings.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate and save
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    logger.success(f"✅ Meta-model trained. RMSE: {rmse:.4f}")

    save_model("meta_model", {
        "model": model,
        "features": list(X.columns),
        "params": {"n_estimators": settings.meta_n_estimators}
    })

    save_data(pd.DataFrame([{
        "model_name": "meta_model",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "rmse": rmse,
        "accuracy": None
    }]), settings.meta_metadata_table)

    logger.success("🚀 Meta model training complete.")

if __name__ == "__main__":
    train_meta_model()
