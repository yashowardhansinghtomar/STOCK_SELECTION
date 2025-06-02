# models/train_entry_exit_model.py

import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from core.logger.logger import logger
from core.model_io import save_model
from core.data_provider.data_provider import load_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features


def parse_exit_config(cfg_str: str) -> tuple:
    try:
        cfg = json.loads(cfg_str)
        exit_rule = cfg.get("exit_rule", {})
        return (
            exit_rule.get("kind", "fixed_pct"),
            exit_rule.get("stop_loss", 0),
            exit_rule.get("take_profit", 0),
            exit_rule.get("trail", 0),
            exit_rule.get("sma_window", 0),
            exit_rule.get("max_holding_days", 0),
        )
    except Exception:
        return ("fixed_pct", 0, 0, 0, 0, 0)


def train_entry_exit_model():
    logger.start("🚀 Training Joint Entry+Exit Model...")

    trades = load_data("paper_trades")
    if trades is None or trades.empty:
        logger.error("❌ No paper trades found.")
        return

    rows = []
    for _, row in trades.iterrows():
        stock = row["stock"]
        ts = pd.to_datetime(row["timestamp"], errors="coerce")
        if pd.isna(ts):
            continue

        feats = enrich_multi_interval_features(stock, ts)
        if feats.empty:
            continue

        entry_signal = int(row.get("profit", 0) > 0)
        kind, sl, tp, trail, sma_win, max_hold = parse_exit_config(row.get("strategy_config", "{}"))

        feats["entry_signal"] = entry_signal
        feats["exit_kind"] = kind
        feats["stop_loss"] = sl
        feats["take_profit"] = tp
        feats["trail"] = trail
        feats["exit_sma_window"] = sma_win
        feats["max_holding_days"] = max_hold

        rows.append(feats)

    if not rows:
        logger.error("❌ No enriched rows found.")
        return

    df = pd.concat(rows, ignore_index=True)

    label_enc = LabelEncoder()
    df["exit_kind_encoded"] = label_enc.fit_transform(df["exit_kind"].astype(str))

    X = df.drop(columns=["stock", "date", "entry_signal", "exit_kind"], errors="ignore")
    y_class = df["entry_signal"]
    y_exit = df[["exit_kind_encoded", "stop_loss", "take_profit", "trail", "exit_sma_window", "max_holding_days"]]

    X_train, X_test, y_class_train, y_class_test, y_exit_train, y_exit_test = train_test_split(
        X, y_class, y_exit, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_class_train)
    preds = clf.predict(X_test)

    logger.success("✅ Entry signal classifier report:")
    logger.info("\n" + classification_report(y_class_test, preds))

    reg = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42))
    reg.fit(X_train, y_exit_train)

    save_model("entry_exit_model", {
        "clf": clf,
        "reg": reg,
        "features": list(X.columns),
        "exit_kind_encoder": label_enc
    })

    logger.success("📦 Entry+Exit model saved as 'entry_exit_model'")


if __name__ == "__main__":
    train_entry_exit_model()
