import pandas as pd
import lightgbm as lgb
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from core.config.config import settings
from core.data_provider.data_provider import load_data, save_data
from core.logger.logger import logger
from pathlib import Path

FEATURE_TABLE = "stock_features_day"
RECS_TABLE = "recommendations"
PRED_TABLE = settings.tables.predictions["filter"]

FEATURE_COLS = [
    "sma_short", "sma_long", "rsi_thresh", "macd", "vwap", "atr_14",
    "bb_width", "macd_histogram", "price_compression", "volatility_10",
    "volume_spike", "vwap_dev", "stock_encoded"
]
LABEL_COL = "label"


def save_model_and_predictions(model, X_val, y_proba, merged):
    preds = X_val.copy()
    preds["score"] = y_proba
    preds["rank"] = preds["score"].rank(pct=True)
    preds["confidence"] = preds["score"]
    preds["decision"] = preds["score"].gt(0.5).map({True: "buy", False: "reject"})
    preds["stock"] = merged.iloc[preds.index]["stock"].values
    preds["date"] = merged.iloc[preds.index]["date"].values

    save_data(
        preds[["date", "stock", "score", "rank", "confidence", "decision"]],
        PRED_TABLE,
        if_exists="replace"
    )

    model_path = Path(settings.model_dir) / f"{settings.model_names['filter']}.lgb"
    joblib.dump(model, model_path)
    logger.success("✅ Filter model trained and saved. Predictions written to DB.")


def train_filter_model():
    recs = load_data(RECS_TABLE)
    feats = load_data(FEATURE_TABLE)

    # Drop rows with bad dates
    recs = recs.dropna(subset=["date"])
    feats = feats.dropna(subset=["date"])

    # Normalize join keys
    recs["stock"] = recs["stock"].str.strip().str.upper()
    feats["stock"] = feats["stock"].str.strip().str.upper()
    recs["date"] = pd.to_datetime(recs["date"]).dt.normalize()
    feats["date"] = pd.to_datetime(feats["date"]).dt.normalize()

    logger.info(f"🧪 RECS: {len(recs)} rows | FEATS: {len(feats)} rows")
    try:
        logger.info(f"📆 RECS date range: {recs['date'].min().date()} → {recs['date'].max().date()}")
        logger.info(f"📆 FEATS date range: {feats['date'].min().date()} → {feats['date'].max().date()}")
    except Exception as e:
        logger.warning(f"⚠️ Could not compute date ranges: {e}")

    if recs.empty:
        logger.error("❌ No recommendations found. Please run bootstrap_filter_training_data.py first.")
        return
    if feats.empty:
        logger.error("❌ No feature data found. Please backfill stock_features_day first.")
        return

    logger.debug(f"📋 RECS columns: {list(recs.columns)}")
    logger.debug(f"📋 FEATS columns: {list(feats.columns)}")

    merged = recs.merge(feats, on=["stock", "date"], how="inner", suffixes=('', '_feat'))
    logger.info(f"🔍 Merged rows: {len(merged)}")

    if merged.empty:
        mismatch = recs.merge(feats, on=["stock", "date"], how="left", indicator=True)
        unmatched = mismatch[mismatch["_merge"] == "left_only"]
        logger.warning(f"⚠️ Mismatched recommendation rows: {len(unmatched)} / {len(recs)}")
        logger.debug(f"📄 Unmatched preview:\n{unmatched[['stock', 'date']].head()}")
        logger.error("❌ Merge produced 0 rows — check for date mismatches or missing symbols.")
        return

    logger.debug(f"📋 Merged columns: {list(merged.columns)}")

    missing_cols = [col for col in FEATURE_COLS + [LABEL_COL] if col not in merged.columns]
    if missing_cols:
        logger.error(f"❌ Missing required columns in merged data: {missing_cols}")
        return

    if merged[LABEL_COL].isnull().all():
        logger.warning("⚠️ Label column is present but all values are null.")

    before_dropna = len(merged)
    merged = merged.dropna(subset=FEATURE_COLS + [LABEL_COL])
    after_dropna = len(merged)
    logger.info(f"📉 Dropped {before_dropna - after_dropna} rows with missing features or labels.")
    logger.info(f"📈 Remaining for training: {after_dropna} rows")

    if len(merged) < 100:
        logger.warning("⚠️ Not enough labeled data to train a robust model. Skipping model training.")
        return

    X = merged[FEATURE_COLS]
    y = merged[LABEL_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    logger.info("📊 Model Performance on Validation Set:")
    logger.info("\n" + classification_report(y_val, y_pred))
    logger.info(f"🔍 ROC AUC: {roc_auc_score(y_val, y_proba):.3f}")

    save_model_and_predictions(model, X_val, y_proba, merged)


if __name__ == "__main__":
    train_filter_model()
