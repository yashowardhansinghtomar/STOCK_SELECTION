from core.model_io import save_model
from core.logger.logger import logger
from core.data_provider.data_provider import load_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.config.config import settings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def train_exit_model():
    logger.start("🚀 Starting Exit Model Training (per interval)...")

    df_trades = load_data(settings.trades_table)
    if df_trades is None or df_trades.empty:
        logger.error("❌ No paper trades found.")
        return

    rows = []
    for _, row in df_trades.iterrows():
        stock = row["stock"]
        timestamp = pd.to_datetime(row["timestamp"], errors="coerce")
        interval = row.get("interval", "day")
        if pd.isna(timestamp):
            continue
        feat = enrich_multi_interval_features(stock, timestamp, intervals=[interval])
        if feat.empty:
            continue
        feat["target"] = int(row.get("profit", 0) > 0)
        feat["interval"] = interval
        rows.append(feat)

    if not rows:
        logger.error("❌ No enriched rows available for training.")
        return

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all.dropna(subset=["target"])
    intervals = df_all["interval"].dropna().unique().tolist()

    for interval in intervals:
        df = df_all[df_all["interval"] == interval].copy()
        if df.empty:
            continue

        features = settings.exit_feature_columns
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.warning(f"⚠️ Skipping interval {interval} — missing: {missing}")
            continue

        X = df[features].fillna(0).replace([float("inf"), float("-inf")], 0)
        y = df["target"]

        logger.info(f"📊 Class distribution for '{interval}': {Counter(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.test_size, stratify=y, random_state=settings.random_state
        )

        clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)

        logger.success(f"✅ Exit Model ({interval}) Accuracy: {acc:.2%}")
        logger.info("\n" + classification_report(y_test, preds))

        model_name = f"{settings.model_names['exit']}_{interval}"
        save_model(model_name, {
            "model": clf,
            "features": list(X.columns)
        }, meta={
            "trained_at": str(pd.Timestamp.now()),
            "interval": interval,
            "accuracy": round(acc, 4),
            "confidence_mean": round(proba.mean(), 4),
            "confidence_std": round(proba.std(), 4),
            "training_rows": len(df),
            "feature_count": len(X.columns),
            "algo": "RandomForestClassifier"
        })

        logger.success(f"📦 Saved: {model_name}")

    logger.success("📁 All exit models saved.")


if __name__ == "__main__":
    train_exit_model()
