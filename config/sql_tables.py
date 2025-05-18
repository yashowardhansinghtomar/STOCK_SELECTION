# config/sql_tables.py

SQL_TABLES = {
    # Core Tables
    "recommendations": "recommendations",
    "open_positions": "open_positions",
    "paper_trades": "paper_trades",
    "training_data": "training_data",

    # Feature + Model Inputs
    "stock_fundamentals": "stock_fundamentals",
    "stock_price_history": "stock_price_history",
    "ml_selected_stocks": "ml_selected_stocks",
    "stock_labels": "stock_labels",

    # Logs & Results
    "backtest_summaries": "backtest_summaries",
    "walkforward_log": "walkforward_log",

    # System Metadata (Optional)
    "model_store": "model_store",
    "json_store": "json_store",
}
