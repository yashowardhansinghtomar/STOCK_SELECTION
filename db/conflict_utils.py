from sqlalchemy.dialects.postgresql import insert
from db.db import engine
from db.models import Base
import pandas as pd

# Define per-table conflict handling rules
CONFLICT_HANDLERS = {
    "stock_fundamentals": (["stock"], "UPDATE"),
    "stock_features": (["stock", "date"], "UPDATE"),
    "stock_price_history": (["symbol", "date", "interval"], "DO NOTHING"),
    "trades": (["symbol", "timestamp"], "DO NOTHING"),
    "paper_trades": (["timestamp", "stock", "action"], "DO NOTHING"),
    "open_positions": (["stock"], "REPLACE"),
    "model_store": (["model_name"], "UPDATE"),
    "recommendations": (["stock", "date"], "REPLACE"), 
    "param_model_predictions": (["date", "stock"], "UPDATE"),
    "filter_model_predictions": (["date", "stock"], "REPLACE"),
    "price_model_predictions": (["date", "stock"], "UPDATE"),
}

Base.metadata.reflect(bind=engine)

def insert_with_conflict_handling(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append",
    chunk_size: int = 1000
) -> None:
    if df.empty:
        return

    conflict_cols, action = CONFLICT_HANDLERS.get(table_name, (None, None))
    if conflict_cols:
        df = df.drop_duplicates(subset=conflict_cols, keep="last")

    table = Base.metadata.tables.get(table_name)
    if table is None:
        raise ValueError(f"Unknown table: {table_name}")

    if "trade_triggered" in df.columns:
        df["trade_triggered"] = df["trade_triggered"].apply(lambda x: int(bool(x)) if pd.notnull(x) else None)
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)

    with engine.begin() as conn:
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            records = chunk_df.to_dict(orient="records")
            stmt = insert(table).values(records)

            if conflict_cols:
                if action == "UPDATE":
                    update_cols = {
                        col: stmt.excluded[col]
                        for col in chunk_df.columns
                        if col not in conflict_cols
                    }
                    stmt = stmt.on_conflict_do_update(
                        index_elements=conflict_cols,
                        set_=update_cols
                    )
                elif action == "DO NOTHING":
                    stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
                elif action == "REPLACE":
                    update_cols = {col: stmt.excluded[col] for col in chunk_df.columns}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=conflict_cols,
                        set_=update_cols
                    )

            conn.execute(stmt)
