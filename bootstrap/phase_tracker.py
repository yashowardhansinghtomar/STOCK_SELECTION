import pandas as pd
import matplotlib.pyplot as plt
import time
from sqlalchemy import create_engine
from datetime import datetime

DATABASE_URL = "postgresql+psycopg2://postgres:0809@localhost:5432/trading_db"
engine = create_engine(DATABASE_URL)

REFRESH_INTERVAL = 15  # seconds

def load_phase_data():
    query = "SELECT * FROM system_phase_history ORDER BY date"
    return pd.read_sql(query, engine)

def plot_phase_dashboard(df):
    plt.clf()
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Phase Step Plot
    axs[0].step(df["date"], df["phase"], where='post', label="Phase", linewidth=2)
    axs[0].set_ylabel("Phase")
    axs[0].set_title("System Phase Over Time")
    axs[0].grid(True)
    axs[0].legend()

    # Epsilon
    axs[1].plot(df["date"], df["epsilon"], marker='o', label="Epsilon (ε)")
    axs[1].set_ylabel("Epsilon")
    axs[1].set_title("Epsilon Decay Over Time")
    axs[1].grid(True)
    axs[1].legend()

    # Real Trades
    axs[2].bar(df["date"], df["real_trade_count"], label="Real Trades")
    axs[2].set_ylabel("Real Trades")
    axs[2].set_xlabel("Date")
    axs[2].set_title("Real Trades Logged Per Day")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.pause(0.1)

if __name__ == "__main__":
    plt.ion()  # Interactive mode ON

    while True:
        try:
            df = load_phase_data()
            if not df.empty:
                plot_phase_dashboard(df)
            else:
                print("No phase data available yet...")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(REFRESH_INTERVAL)
