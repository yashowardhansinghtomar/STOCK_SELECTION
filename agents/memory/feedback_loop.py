# agents/memory/feedback_loop.py

import pandas as pd
import json
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import load_data, save_data
from core.config.config import settings, FeatureGroupConfig
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from db.replay_buffer_sql import ReplayBuffer
from pytz import timezone

IST = timezone("Asia/Kolkata")

def parse_exit_field(json_str, key):
    try:
        if isinstance(json_str, str):
            cfg = json.loads(json_str)
        elif isinstance(json_str, dict):
            cfg = json_str
        else:
            return None
        return cfg.get("exit_rule", {}).get(key)
    except Exception:
        return None

def update_training_data():
    logger.start("🧠 Updating training data via feedback loop…")

    today = pd.to_datetime(get_simulation_date()).astimezone(IST)
    today_date = today.date()

    trades = load_data(settings.tables.trades)
    if trades is None or trades.empty:
        logger.info("📬 No paper trades to update.")
        return

    trades["timestamp"] = pd.to_datetime(trades["timestamp"], errors="coerce")
    if trades["timestamp"].dt.tz is None:
        trades["timestamp"] = trades["timestamp"].dt.tz_localize("Asia/Kolkata")
    else:
        trades["timestamp"] = trades["timestamp"].dt.tz_convert("Asia/Kolkata")

    trades_today = trades[trades["timestamp"].dt.date == today_date].copy()
    if trades_today.empty:
        logger.info("📬 No trades executed today.")
        return

    logger.info(f"📦 Found {len(trades_today)} trades. Enriching features...")

    buffer = ReplayBuffer()
    rows = []
    for _, row in trades_today.iterrows():
        stock = row["stock"].upper()
        ts = row["timestamp"]
        interval = row.get("interval", "day")
        strategy_cfg = row.get("strategy_config", {})

        if isinstance(strategy_cfg, str):
            try:
                strategy_cfg = json.loads(strategy_cfg)
            except json.JSONDecodeError:
                strategy_cfg = {}

        feature_date = ts.normalize() - pd.Timedelta(days=1)
        enriched = enrich_multi_interval_features(stock, feature_date, intervals=[interval])
        if enriched.empty:
            continue

        enriched["stock"] = stock
        enriched["timestamp"] = ts
        enriched["profit"] = row.get("profit", 0)
        enriched["exit_price"] = row.get("price", None)
        enriched["strategy_config"] = strategy_cfg
        enriched["interval"] = interval
        enriched["target"] = int(enriched["profit"] > 0)
        enriched["signal_reason"] = row.get("signal_reason", None)
        rows.append(enriched)
        state = enriched.drop(columns=["timestamp", "stock", "target"], errors="ignore").iloc[0].to_dict()
        reward = float(row.get("profit", 0))

        # --- Reward Attribution ---
        missed_profit = compute_missed_profit(row)
        hold_penalty = compute_holding_penalty(row)
        total_reward = reward + missed_profit - hold_penalty
        done = True  # since trade is closed

        buffer.add_episode(
            stock=stock,
            timestamp=ts,
            state=state,
            action=1,
            reward=total_reward,
            next_state=state,
            done=done,
            interval=interval,
            strategy_config=strategy_cfg,
            extra_fields={
                "missed_profit": missed_profit,
                "hold_duration_penalty": hold_penalty,
                "reason": row.get("signal_reason", "exit")
            }
        )

    if not rows:
        logger.warnings("⚠️ No matching features for today’s trades.")
        return

    df = pd.concat(rows, ignore_index=True)

    # Flatten out strategy config exit fields
    df["exit_kind"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "kind"))
    df["stop_loss"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "stop_loss"))
    df["take_profit"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "take_profit"))
    df["trail"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "trail"))
    df["exit_sma_window"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "sma_window"))
    df["max_holding_days"] = df["strategy_config"].apply(lambda x: parse_exit_field(x, "max_holding_days"))

    training_df = df[FeatureGroupConfig.training_columns + ["interval"]].copy()
    training_df = training_df.dropna(subset=["target"])

    if training_df.empty:
        logger.warning("⚠️ No valid training rows to insert.")
        return

    # Add required fields
    training_df["entry_date"] = training_df["timestamp"].dt.date
    training_df["run_timestamp"] = today

    # Convert raw features to a JSON dict (excluding meta columns)
    drop_cols = [
        "stock", "timestamp", "target", "entry_date", "run_timestamp", "interval", "label"
    ]
    training_df["features"] = training_df.drop(columns=drop_cols, errors="ignore").to_dict(orient="records")

    # Add label field (same as target)
    training_df["label"] = training_df["target"]

    # Keep only the required columns
    final_df = training_df[["stock", "entry_date", "features", "label", "run_timestamp"]].copy()

    save_data(final_df, settings.tables.training_data)
    logger.success(f"✅ Inserted {len(final_df)} new training rows.")


def compute_missed_profit(trade):
    try:
        entry_price = trade.get("entry_price")
        high_price = trade.get("high")  # during holding period
        if entry_price is not None and high_price is not None:
            missed = max(0, high_price - max(entry_price, trade.get("price", 0)))
            return round(missed, 2)
    except Exception:
        pass
    return 0.0

def compute_holding_penalty(trade):
    try:
        entry_ts = pd.to_datetime(trade.get("entry_timestamp"))
        exit_ts = pd.to_datetime(trade.get("timestamp"))
        if entry_ts and exit_ts:
            hold_days = (exit_ts - entry_ts).days
            return round(0.1 * hold_days, 2)  # Flat penalty per day
    except Exception:
        pass
    return 0.0
