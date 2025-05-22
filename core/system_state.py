# core/system_state.py

import pandas as pd
import json
from core.logger import logger
from core.model_io import load_model
from core.config import settings
from core.data_provider import load_data
from datetime import datetime
from core.time_context import get_simulation_date


def get_market_data_state():
    df = load_data(settings.price_history_table)
    today = pd.to_datetime(get_simulation_date()).date()
    if df is None or df.empty:
        return {"status": "ERROR", "message": "No price history found."}
    missing = df[df.date < pd.Timestamp(today)].symbol.unique().tolist()
    return {
        "latest_date": str(df.date.max().date()),
        "symbols_missing_today": list(missing),
        "status": "OK" if not missing else "WARNING"
    }


def get_feature_state():
    status = {}
    for interval in ["day", "15m", "60m"]:
        table = settings.interval_feature_table_map.get(interval, f"stock_features_{interval}")
        df = load_data(table)
        if df is None or df.empty:
            status[interval] = {"status": "ERROR", "message": "Missing table or empty."}
        else:
            latest = pd.to_datetime(df.date).max().date()
            status[interval] = {"count": len(df), "latest": str(latest), "status": "OK"}
    return status


def get_model_state():
    models = ["filter_model", "param_model", "exit_classifier", "meta_model", "entry_exit_model"]
    output = {}
    for m in models:
        try:
            obj = load_model(m)
            output[m] = {"loaded": True, "details": str(type(obj))}
        except Exception as e:
            output[m] = {"loaded": False, "error": str(e)}
    return output


def get_planner_state():
    df = load_data(settings.recommendations_table)
    today = pd.to_datetime(get_simulation_date()).date()
    if df is None or df.empty:
        return {"status": "ERROR", "message": "No recommendations data."}
    today_signals = df[pd.to_datetime(df.date).dt.date == today]
    return {
        "status": "OK" if not today_signals.empty else "WARNING",
        "signals_today": len(today_signals),
        "top_n": settings.top_n
    }


def get_execution_state():
    trades = load_data(settings.trades_table)
    open_pos = load_data(settings.open_positions_table)
    today = pd.to_datetime(get_simulation_date()).date()
    count_today = 0
    if trades is not None and not trades.empty:
        count_today = len(trades[pd.to_datetime(trades.timestamp).dt.date == today])
    return {
        "today_trades": count_today,
        "open_positions": 0 if open_pos is None else len(open_pos),
        "status": "OK"
    }


def build_system_state():
    state = {
        "date": str(get_simulation_date()),
        "market_data": get_market_data_state(),
        "features": get_feature_state(),
        "models": get_model_state(),
        "planner": get_planner_state(),
        "execution": get_execution_state(),
    }
    return state


if __name__ == "__main__":
    system_state = build_system_state()
    print(json.dumps(system_state, indent=2))
