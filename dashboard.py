import streamlit as st
import pandas as pd
import sqlalchemy
import json
from datetime import datetime
import numpy as np

# --- Config ---
DB_URL = "postgresql+psycopg2://postgres:0809@localhost:5432/trading_db"
engine = sqlalchemy.create_engine(DB_URL)

# --- Title ---
st.set_page_config(page_title="O.D.I.N. Dashboard", layout="wide")
st.title("\U0001F4CA O.D.I.N. Live Dashboard")

# --- Helper to run SQL ---
def run_query(sql):
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)

dashboard_stats = {}

# --- Phase Overview ---
st.header("\U0001F6A6 Phase & Policy Overview")
phase_df = run_query("""
    SELECT * FROM system_phase_history
    ORDER BY date DESC LIMIT 1
""")
if not phase_df.empty:
    row = phase_df.iloc[0]
    dashboard_stats["phase"] = int(row['phase'])
    dashboard_stats["real_trade_count"] = int(row['real_trade_count'])
    dashboard_stats["avg_reward_ok"] = bool(row['converged'])
    dashboard_stats["epsilon"] = float(row['epsilon'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Phase", row['phase'])
    col2.metric("Real Trades", row['real_trade_count'])
    col3.metric("Avg Reward OK", str(row['converged']))
    col4.metric("Epsilon", round(row['epsilon'], 3))
else:
    st.warning("No phase history available.")

# --- Replay Buffer Stats ---
st.header("\U0001F9E0 Replay Buffer Stats")
buffer_size = run_query("SELECT COUNT(*) AS count FROM rl_replay_buffer")
real_trades = run_query("SELECT COUNT(*) AS real FROM rl_replay_buffer WHERE reward IS NOT NULL")
counts_by_stock = run_query("""
    SELECT stock, COUNT(*) AS count
    FROM rl_replay_buffer
    GROUP BY stock ORDER BY count DESC LIMIT 10
""")
dashboard_stats["buffer_size"] = int(buffer_size.iloc[0]["count"])
dashboard_stats["real_trades"] = int(real_trades.iloc[0]["real"])
dashboard_stats["top_stocks"] = counts_by_stock.to_dict(orient="records")

col1, col2 = st.columns(2)
col1.metric("Buffer Size", buffer_size.iloc[0]["count"])
col2.metric("Real Trades", real_trades.iloc[0]["real"])
st.subheader("Top 10 Most Traded Stocks")
st.dataframe(counts_by_stock)

# --- Reward Trend ---
st.header("\U0001F4C8 Reward Trend")
reward_history = run_query("""
    SELECT date, AVG(reward) AS avg_reward
    FROM rl_replay_buffer
    WHERE reward IS NOT NULL
    GROUP BY date ORDER BY date
""")
if not reward_history.empty:
    reward_history["avg_reward"] = reward_history["avg_reward"].astype(float)
    dashboard_stats["reward_trend"] = reward_history.tail(10).to_dict(orient="records")
    dashboard_stats["rolling_avg_reward_3"] = reward_history["avg_reward"].rolling(3).mean().tail(1).item()
    dashboard_stats["rolling_avg_reward_5"] = reward_history["avg_reward"].rolling(5).mean().tail(1).item()
    dashboard_stats["rolling_avg_reward_10"] = reward_history["avg_reward"].rolling(10).mean().tail(1).item()
    rolling = reward_history["avg_reward"].rolling(5).mean()
    st.line_chart(pd.DataFrame({"avg_reward": reward_history["avg_reward"], "rolling": rolling}).set_index(reward_history["date"]))
else:
    st.info("No reward data available.")

# --- Win Rate by Symbol ---
winrate_query = run_query("""
    SELECT stock, 
           COUNT(*) FILTER (WHERE reward > 0) * 100.0 / COUNT(*) AS win_rate
    FROM rl_replay_buffer
    WHERE reward IS NOT NULL
    GROUP BY stock ORDER BY win_rate DESC LIMIT 10
""")
if not winrate_query.empty:
    dashboard_stats["top_winrate"] = winrate_query.to_dict(orient="records")
    st.subheader("\U0001F4C9 Top 10 Symbols by Win Rate")
    st.dataframe(winrate_query)

# --- Unique Symbols Traded ---
unique_stocks_query = run_query("SELECT COUNT(DISTINCT stock) AS unique_count FROM rl_replay_buffer")
dashboard_stats["total_unique_stocks_traded"] = int(unique_stocks_query.iloc[0]["unique_count"])
st.metric("Total Unique Stocks Traded", dashboard_stats["total_unique_stocks_traded"])

# # --- Exit Mode Summary ---
# exit_mode_summary = run_query("""
#     SELECT meta ->> 'exit_mode' AS exit_mode, COUNT(*) AS count
#     FROM rl_replay_buffer
#     WHERE meta IS NOT NULL AND meta ->> 'exit_mode' IS NOT NULL
#     GROUP BY exit_mode ORDER BY count DESC
# """)
# if not exit_mode_summary.empty:
#     dashboard_stats["exit_mode_summary"] = exit_mode_summary.to_dict(orient="records")
#     st.subheader("\U0001F6A8 Exit Mode Breakdown")
#     st.dataframe(exit_mode_summary)

# --- Today's Trades ---
st.header("\U0001F4C5 Today's Trades")
today = datetime.now().date()
today_trades = run_query(f"""
    SELECT * FROM rl_replay_buffer WHERE date = '{today}'
    ORDER BY inserted_at DESC
""")
dashboard_stats["today_trade_count"] = len(today_trades)
if not today_trades.empty:
    st.dataframe(today_trades)
else:
    st.info("No trades recorded today.")

# --- Export to JSON for ChatGPT / Review ---
dashboard_stats["timestamp"] = datetime.now().isoformat()

def safe_convert(obj):
    try:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: safe_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_convert(v) for v in obj]
        else:
            return str(obj)  # fallback: stringify unknown objects
    except Exception as e:
        return f"Unserializable: {type(obj).__name__}"

with open("odin_stats_export.json", "w") as f:
    json.dump(safe_convert(dashboard_stats), f, indent=4)

st.success("\U0001F4E4 Exported dashboard stats to odin_stats_export.json")
