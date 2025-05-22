# models/train_exit_model.py

from core.model_io import save_model
from core.logger import logger
from core.data_provider import load_data
from core.feature_enricher_multi import enrich_multi_interval_features
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from core.config import get_feature_columns


def train_exit_model():
    logger.start("🚀 Starting Exit Model Training...")

    df_trades = load_data("paper_trades")
    if df_trades is None or df_trades.empty:
        logger.error("❌ No paper trades found.")
        return

    rows = []
    for _, row in df_trades.iterrows():
        stock = row["stock"]
        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(timestamp):
            continue
        feat = enrich_multi_interval_features(stock, timestamp)
        if feat.empty:
            continue
        feat["target"] = int(row.get("profit", 0) > 0)
        rows.append(feat)

    if not rows:
        logger.error("❌ No enriched rows available for training.")
        return

    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["target"])

    X = df[get_feature_columns("day")]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)

    logger.success(f"✅ Exit Model Accuracy: {acc:.2%}")
    logger.info("\n" + classification_report(y_test, preds))

    save_model("exit_classifier", (clf, list(X.columns)), meta={
        "trained_at": str(pd.Timestamp.now()),
        "accuracy": round(acc, 4),
        "confidence_mean": round(proba.mean(), 4),
        "confidence_std": round(proba.std(), 4),
        "training_rows": len(df),
        "feature_count": len(X.columns),
        "algo": "RandomForestClassifier"
    })

    logger.info("📦 Exit model and metadata saved to model_store.")


if __name__ == "__main__":
    train_exit_model()
