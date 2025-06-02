# dashboards/system_dashboard.py

import streamlit as st
import pandas as pd
from db.postgres_manager import run_query
from datetime import datetime, timedelta
from reports.weekly_snapshot import compute_snapshot

st.set_page_config(layout="wide")
st.title("📊 O.D.I.N. System Dashboard")

# Load system log
def load_system_log():
    query = """
    SELECT * FROM system_log
    ORDER BY timestamp DESC
    LIMIT 1000
    """
    return run_query(query)

# Load paper trades
def load_trade_summary():
    query = """
    SELECT stock, action, price, interval, strategy_config, imported_at, signal_reason
    FROM paper_trades
    ORDER BY imported_at DESC
    LIMIT 1000
    """
    return run_query(query)

# Tabs
log_tab, trade_tab, metrics_tab = st.tabs(["🧠 System Logs", "📈 Recent Trades", "📅 Snapshot"])

with log_tab:
    logs = load_system_log()
    if logs is not None and not logs.empty:
        st.subheader("Filter logs")
        agent_filter = st.multiselect("Agent", options=logs["agent"].unique(), default=list(logs["agent"].unique()))
        action_filter = st.multiselect("Action", options=logs["action"].unique(), default=list(logs["action"].unique()))
        date_range = st.date_input("Date range", [])

        filtered = logs[logs["agent"].isin(agent_filter) & logs["action"].isin(action_filter)]
        if len(date_range) == 2:
            filtered = filtered[(filtered["timestamp"].dt.date >= date_range[0]) & (filtered["timestamp"].dt.date <= date_range[1])]

        st.dataframe(filtered)
        st.success(f"Showing {len(filtered)} filtered logs.")
    else:
        st.log_warning("No system logs found.")

with trade_tab:
    trades = load_trade_summary()
    if trades is not None and not trades.empty:
        st.subheader("Filter trades")
        interval_filter = st.multiselect("Interval", trades["interval"].unique(), default=list(trades["interval"].unique()))
        reason_filter = st.multiselect("Signal Reason", trades["signal_reason"].unique(), default=list(trades["signal_reason"].unique()))
        trades = trades[trades["interval"].isin(interval_filter) & trades["signal_reason"].isin(reason_filter)]

        st.dataframe(trades)
        st.success(f"Showing {len(trades)} filtered trades.")
    else:
        st.log_warning("No paper trades found.")

with metrics_tab:
    st.subheader("Daily Snapshot")
    default_start = datetime.now().date() - timedelta(days=1)
    default_end = datetime.now().date()
    start_date = st.date_input("Start date", default_start)
    end_date = st.date_input("End date", default_end)

    snapshot = compute_snapshot(start_date, end_date)
    if snapshot:
        st.json(snapshot)
    else:
        st.log_warning("No snapshot available.")
