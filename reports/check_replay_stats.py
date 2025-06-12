import pandas as pd
from sqlalchemy import create_engine

# Update with your actual credentials
DB_URL = "postgresql+psycopg2://postgres:0809@localhost:5432/trading_db"
engine = create_engine(DB_URL)

# Get last 500 replay trades with reward
df = pd.read_sql("""
    SELECT date, stock, reward, features
    FROM rl_replay_buffer
    WHERE reward IS NOT NULL
    ORDER BY date DESC
    LIMIT 500
""", engine)

# Compute core stats
print("\n🎯 Replay Buffer Summary (Last 500 Trades):")
print(f"- Trades Logged: {len(df)}")
print(f"- Avg Reward: {df['reward'].mean():.4f}")
print(f"- Std Dev: {df['reward'].std():.4f}")
print(f"- Date Range: {df['date'].min()} → {df['date'].max()}")

# Extract exploration type from JSON
try:
    df["exploration_type"] = df["features"].apply(lambda x: eval(x).get("exploration_type", "unknown"))
    print("\n🔍 Exploration Type Distribution:")
    print(df["exploration_type"].value_counts())
except Exception as e:
    print("⚠️ Could not extract exploration types:", e)
