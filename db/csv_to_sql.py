# db/csv_to_sql.py
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from db.db_router import insert_dataframe

csv_map = {
    "recommendations": "project_data/results/weekly_recommendations.csv",
    "trades": "project_data/results/paper_trades.csv",
    "open_positions": "project_data/results/open_positions.csv",
    "training_data": "project_data/models/training_data.csv",
    "fundamentals": "project_data/results/stock_fundamentals.csv",
    "stock_labels": "project_data/results/stock_labels.csv",
    "ml_selected_stocks": "project_data/results/ml_selected_stocks.csv"
}

def csv_to_sql():
    for table, path in csv_map.items():
        file = Path(path)
        if not file.exists():
            print(f"⚠️ File missing: {path} → skipping table '{table}'")
            continue
        try:
            df = pd.read_csv(path)
            insert_dataframe(df, table)
        except Exception as e:
            print(f"❌ Failed to import {path}: {e}")

if __name__ == "__main__":
    csv_to_sql()
