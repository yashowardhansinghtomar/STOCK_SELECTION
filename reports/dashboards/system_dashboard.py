import streamlit as st
import pandas as pd
from datetime import datetime
import logging
from db.postgres_manager import run_query

# Enable debug logs in terminal
logging.basicConfig(level=logging.DEBUG)

st.set_page_config(page_title="🧠 O.D.I.N. Control Tower", layout="wide")
st.title("🧠 O.D.I.N. Control Tower Dashboard")

# === SECTION 1: SYSTEM STATUS ===
st.header("🔧 System Status")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Today", datetime.now().strftime("%Y-%m-%d"))
with col2:
    st.metric("Mode", "Simulation")
with col3:
    st.metric("Interval", "Day")

# === SECTION 2: AGENT STATUS ===
st.header("🤖 Agent Run Summary")

def load_agent_status():
    query = """
    SELECT agent_name AS Agent, role AS Role,
           status || ' ' || to_char(run_time, 'HH24:MI') AS "Last Run",
           summary AS "Key Action"
    FROM system_log
    WHERE log_date = CURRENT_DATE
    ORDER BY run_time DESC;
    """
    try:
        return run_query(query)
    except Exception as e:
        st.subheader("⚠️ Agent Status Error")
        st.exception(e)
        return pd.DataFrame()

agent_df = load_agent_status()
st.dataframe(agent_df, use_container_width=True)

# === SECTION 3: TRADES TODAY ===
st.header("📊 Trades Summary")

def load_today_trades():
    query = """
    SELECT stock, action, strategy_type, confidence, pnl
    FROM paper_trades
    WHERE trade_date = CURRENT_DATE
    ORDER BY confidence DESC;
    """
    try:
        df = run_query(query)
        df.columns = ["Stock", "Action", "Type", "Confidence", "P&L"]
        return df
    except Exception as e:
        st.subheader("⚠️ Trade Data Error")
        st.exception(e)
        return pd.DataFrame(columns=["Stock", "Action", "Type", "Confidence", "P&L"])

trades_df = load_today_trades()
st.dataframe(trades_df, use_container_width=True)

# === SECTION 4: FEEDBACK & RETRAINING ===
st.header("♻️ Feedback & Learning")

def load_training_data_count():
    query = "SELECT COUNT(*) FROM training_data WHERE DATE(added_at) = CURRENT_DATE;"
    try:
        df = run_query(query)
        return int(df.iloc[0, 0])
    except Exception as e:
        st.subheader("⚠️ Training Count Error")
        st.exception(e)
        return "-"

def load_retrained_models():
    query = "SELECT DISTINCT base_name FROM model_store WHERE DATE(created_at) = CURRENT_DATE;"
    try:
        df = run_query(query)
        models = df["base_name"].tolist()
        return ", ".join(models) if models else "None"
    except Exception as e:
        st.subheader("⚠️ Model Load Error")
        st.exception(e)
        return "-"

def load_feedback_status():
    query = """
    SELECT to_char(run_time, 'HH24:MI') AS last_run
    FROM system_logs
    WHERE agent_name = 'O.R.A.C.L.E.' AND action = 'feedback_loop'
    ORDER BY run_time DESC
    LIMIT 1;
    """
    try:
        df = run_query(query)
        return f"✅ {df.iloc[0]['last_run']}" if not df.empty else "❌ Not run"
    except Exception as e:
        st.subheader("⚠️ Feedback Status Error")
        st.exception(e)
        return "-"

col1, col2, col3 = st.columns(3)
col1.metric("New Training Data", load_training_data_count())
col2.metric("Retrained Models", load_retrained_models())
col3.metric("Feedback Loop", load_feedback_status())

st.caption("© O.D.I.N. Autonomous Trading System — Real-Time Status Monitor")
