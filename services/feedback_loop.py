import pandas as pd
import json
from core.logger import logger
from core.time_context import get_simulation_date
from core.data_provider import load_data, save_data
from core.config import settings
from core.feature_enricher_multi import enrich_multi_interval_features
from replay_buffer import ReplayBuffer  # 🆕 Add this
from pytz import timezone

IST = timezone("Asia/Kolkata")

def update_training_data():
    logger.start("🧠 Updating training data via feedback loop…")

    today = pd.to_datetime(get_simulation_date()).astimezone(IST)
    today_date = today.date()

    trades = load_data(settings.trades_table)
    if trades is None or trades.empty:
        logger.info("📬 No paper trades to update.")
        return

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    trades["timestamp"] = trades["timestamp"].dt.tz_localize("Asia/Kolkata") if trades["timestamp"].dt.tz is None else trades["timestamp"].dt.tz_convert("Asia/Kolkata")

    trades_today = trades[trades["timestamp"].dt.date == today_date].copy()
    if trades_today.empty:
        logger.info("📬 No trades executed today.")
        return

    logger.info(f"📦 Found {len(trades_today)} trades. Enriching features...")

    buffer = ReplayBuffer()  # 🆕 create buffer
    rows = []
    for _, row in trades_today.iterrows():
        stock = row["stock"].upper()
        ts = row["timestamp"]
        feature_date = ts.normalize() - pd.Timedelta(days=1)

        enriched = enrich_multi_interval_features(stock, feature_date)
        if enriched.empty:
            continue

        enriched["stock"] = stock
        enriched["timestamp"] = ts
        enriched["profit"] = row.get("profit", 0)
        enriched["strategy_config"] = row.get("strategy_config", "")
        enriched["interval"] = row.get("interval", "day")
        enriched["target"] = int(enriched["profit"] > 0)
        rows.append(enriched)

        # 🧠 Add to replay buffer
        state = enriched.drop(columns=["timestamp", "stock", "target"], errors="ignore").iloc[0].tolist()
        reward = row.get("profit", 0)
        done = True
        buffer.add_episode(stock, ts, state, action=1, reward=reward, next_state=state, done=done)

    if not rows:
        logger.warning("⚠️ No matching features for today’s trades.")
        return

    df = pd.concat(rows, ignore_index=True)

    def parse_exit_field(json_str, key):
        try:
            cfg = json.loads(json_str)
            return cfg.get("exit_rule", {}).get(key)
        except Exception:
            return None

    df["exit_kind"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "kind"))
    df["stop_loss"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "stop_loss"))
    df["take_profit"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "take_profit"))
    df["trail"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "trail"))
    df["exit_sma_window"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "sma_window"))
    df["max_holding_days"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "max_holding_days"))

    training_df = df[settings.training_columns + ["interval"]].copy()
    training_df = training_df.dropna(subset=["target"])

    if training_df.empty:
        logger.warning("⚠️ No valid training rows to insert.")
        return

    save_data(training_df, settings.training_data_table)
    logger.success(f"✅ Inserted {len(training_df)} new training rows.")
