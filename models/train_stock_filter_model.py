# models/train_stock_filter_model.py
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from core.data_provider import load_data
from core.model_io import save_model
from core.logger import logger
from core.config import settings

def train_stock_filter_model(n_trials: int = 30):
    """
    Train a RandomForest-based stock filter model using features.
    """
    df = load_data("training_data")
    if df is None or df.empty:
        logger.error("❌ No training data found. Cannot train filter model.")
        return

    if settings.use_fundamentals:
        required_cols = ["pe_ratio", "debt_to_equity", "roe", "earnings_growth", "market_cap"]
    else:
        required_cols = [c for c in df.columns if c not in ["stock", "date", "target"]
                         and not any(c in f for f in ["pe_ratio", "debt", "roe", "market_cap", "earnings"])]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns: {missing}")
        return

    X = df[required_cols].fillna(0)
    y = df["target"]

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return accuracy_score(y_val, model.predict(X_val))

    logger.info(f"🔍 Starting HPO for {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logger.success(f"✅ Best HPO params: {best_params}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    save_model("filter_model", {"model": model, "features": required_cols, "params": best_params})

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.success(f"✅ Filter model trained. Test Accuracy: {acc:.4f}")
    logger.info("\n" + classification_report(y_test, preds))

if __name__ == "__main__":
    train_stock_filter_model()
