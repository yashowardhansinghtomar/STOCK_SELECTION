# models/train_exit_model.py

from core.model_io import save_model
from core.logger import logger
from core.data_provider import load_data, save_data  # ✅ updated
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_exit_model():
    logger.start("🚀 Starting Exit Model Training...")

    # ✅ Load from SQL table
    df = load_data("paper_trades")
    if df is None or df.empty or "good_exit" not in df.columns:
        logger.error("❌ Exit training data is empty or missing 'good_exit'.")
        return

    # Drop unnecessary columns
    X = df.drop(columns=["stock", "entry_date", "exit_date", "entry_price", "exit_price", "return_%", "good_exit"], errors="ignore")
    y = df["good_exit"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    logger.success(f"✅ Exit Model Accuracy: {acc:.2%}")
    logger.info("\n" + classification_report(y_test, preds))

    # ✅ Save model to model_store
    save_model("exit_classifier", (clf, list(X.columns)))

    # ✅ Save metadata to SQL
    metadata = {
        "model_name": "exit_classifier",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "accuracy": acc,
        "confidence_mean": round(proba.mean(), 4),
        "confidence_std": round(proba.std(), 4),
        "training_rows": len(df),
        "feature_count": len(X.columns)
    }
    save_data(pd.DataFrame([metadata]), "model_metadata")  # ✅ updated
    logger.info(f"🧠 Metadata saved to table: model_metadata")

if __name__ == "__main__":
    train_exit_model()
