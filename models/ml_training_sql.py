# models/ml_training_sql.py
from core.model_io import save_model
from core.logger import logger
from core.data_provider import load_data, save_data
from core.config import settings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd

def train_meta_model():
    """
    Train a meta-model (regressor) on combined features.
    Uses settings for split and hyperparams.
    """
    logger.start("🚀 Starting Meta Model Training...")

    # Load training data
    df = load_data(settings.meta_training_table)
    if df is None or df.empty:
        logger.error("❌ No meta training data found. Aborting.")
        return

    # Encode stock labels if present
    if "stock" in df.columns:
        le = LabelEncoder()
        df["stock_encoded"] = le.fit_transform(df["stock"].astype(str))

    X = df.drop(columns=[col for col in ["target", "date", "stock"] if col in df.columns])
    y = df["target"]

    # Train/test split using settings
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.test_size,
        random_state=settings.random_state
    )

    # Model init with settings-backed hyperparameters
    n_est = getattr(settings, 'meta_n_estimators', 200)
    model = RandomForestRegressor(
        n_estimators=n_est,
        random_state=settings.random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    logger.success(f"✅ Meta-model trained. RMSE: {rmse:.4f}")

    # Save model + feature list
    save_model(
        "meta_model",
        {"model": model, "features": list(X.columns), "params": {"n_estimators": n_est}}
    )

    # Persist metadata
    meta = {
        "model_name": "meta_model",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "rmse": rmse,
        "accuracy": None
    }
    save_data(pd.DataFrame([meta]), settings.meta_metadata_table)

    logger.success("🚀 Meta model training complete.")

if __name__ == "__main__":
    train_meta_model()
