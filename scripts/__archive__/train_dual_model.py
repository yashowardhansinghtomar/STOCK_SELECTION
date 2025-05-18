from core.model_io import load_model, save_model
from core.logger import logger
# train_dual_model.py
import os
import json
import pandas as pd
import pickle
from config.paths import PATHS
from utils.file_io import load_dataframe

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, classification_report

def train_dual_models():
    df = load_dataframe(PATHS["training_data"])
    if df.empty or 'total_return' not in df or 'trade_triggered' not in df:
        logger.warning("⚠️ No valid data found.")
        return

    features = df.drop(columns=[
        "date", "stock", "total_return", "avg_trade_return", "max_drawdown", "trade_count"
    ], errors='ignore')

    for col in features.select_dtypes(include=['object']).columns:
        features[col] = LabelEncoder().fit_transform(features[col].astype(str))

    labels_trigger = df["trade_triggered"]
    labels_return = df[df.trade_triggered == 1]["total_return"]

    metadata = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "training_rows": len(df),
    }

    # Classifier
    X_train, X_test, y_train, y_test = train_test_split(features, labels_trigger, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    clf_score = clf.score(X_test, y_test)
    metadata["classifier_score"] = clf_score

    save_model(PATHS["trade_classifier"], PATHS["trade_classifier"])
    logger.success(f"✅ Classifier saved to {PATHS['trade_classifier']} (Accuracy: {clf_score:.2%})")

    # Regressor
    X_ret = features[df.trade_triggered == 1]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_ret, labels_return, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_r, y_train_r)
    rmse = mean_squared_error(y_test_r, reg.predict(X_test_r)) ** 0.5
    metadata["regressor_rmse"] = rmse

    save_model(PATHS["return_regressor"], PATHS["return_regressor"])
    logger.success(f"✅ Regressor saved to {PATHS['return_regressor']} (RMSE: {rmse:.4f})")

    # Save metadata
    meta_path = PATHS["trade_classifier"].parent / f"model_metadata_{metadata['date'].replace(':', '-')}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"🧠 Metadata saved to {meta_path}")

if __name__ == "__main__":
    train_dual_models()
