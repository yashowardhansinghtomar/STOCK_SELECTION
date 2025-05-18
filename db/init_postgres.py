# db/init_postgres.py

from db.postgres_manager import execute_raw_sql, run_query

def table_exists(table_name):
    result = run_query("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_name = :table_name
    );
    """, params={"table_name": table_name})
    return result[0][0]



def init_postgres():
    print("🚀 Initializing PostgreSQL schema...\n")

    tables_to_create = {
        "stock_fundamentals": """
            CREATE TABLE IF NOT EXISTS stock_fundamentals (
                stock TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                pe_ratio FLOAT,
                pb_ratio FLOAT,
                roe FLOAT,
                debt_to_equity FLOAT,
                market_cap BIGINT,
                imported_at TIMESTAMP DEFAULT now()
            );
        """,
        "stock_features": """
            CREATE TABLE IF NOT EXISTS stock_features (
                stock TEXT NOT NULL,
                date DATE NOT NULL,
                sma_short FLOAT,
                sma_long FLOAT,
                rsi_thresh FLOAT,
                stock_encoded INT,
                PRIMARY KEY (stock, date)
            );
        """,
        "stock_price_history": """
            CREATE TABLE IF NOT EXISTS stock_price_history (
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT,
                PRIMARY KEY (symbol, date)
            );
        """,
        "ml_selected_stocks": """
            CREATE TABLE IF NOT EXISTS ml_selected_stocks (
                stock TEXT PRIMARY KEY,
                imported_at TIMESTAMP DEFAULT now()
            );
        """,
        "recommendations": """
            CREATE TABLE IF NOT EXISTS recommendations (
                date DATE NOT NULL,
                stock TEXT NOT NULL,
                sma_short FLOAT,
                sma_long FLOAT,
                rsi_thresh FLOAT,
                predicted_return FLOAT,
                trade_triggered INT DEFAULT 1,
                PRIMARY KEY (date, stock)
            );
        """,
        "open_positions": """
            CREATE TABLE IF NOT EXISTS open_positions (
                stock TEXT PRIMARY KEY,
                entry_price FLOAT,
                entry_date DATE,
                sma_short FLOAT,
                sma_long FLOAT,
                rsi_thresh FLOAT
            );
        """,
        "closed_trades": """
            CREATE TABLE IF NOT EXISTS closed_trades (
                stock TEXT,
                entry_price FLOAT,
                exit_price FLOAT,
                entry_date DATE,
                exit_date DATE,
                sma_short FLOAT,
                sma_long FLOAT,
                rsi_thresh FLOAT,
                PRIMARY KEY (stock, entry_date)
            );
        """,
        "trades": """
            CREATE TABLE IF NOT EXISTS trades (
                timestamp TIMESTAMP NOT NULL,
                stock TEXT NOT NULL,
                action TEXT NOT NULL,  -- 'buy' or 'sell'
                price FLOAT,
                strategy_config TEXT,
                signal_reason TEXT,
                source TEXT,
                imported_at TIMESTAMP DEFAULT now()
            );
        """,
        "paper_trades": """
            CREATE TABLE IF NOT EXISTS paper_trades (
                timestamp TIMESTAMP NOT NULL,
                stock TEXT NOT NULL,
                action TEXT NOT NULL,
                price FLOAT,
                quantity INT,
                strategy_config TEXT,
                signal_reason TEXT,
                source TEXT,
                imported_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (timestamp, stock, action)
            );
        """,
        "training_data": """
            CREATE TABLE IF NOT EXISTS training_data (
                stock TEXT NOT NULL,
                entry_date DATE NOT NULL,
                features JSONB,
                label FLOAT,
                PRIMARY KEY (stock, entry_date)
            );
        """
        """

        CREATE TABLE IF NOT EXISTS filter_model_predictions (
            date DATE NOT NULL,
            stock TEXT NOT NULL,
            score FLOAT,
            rank INT,
            confidence FLOAT,
            decision TEXT,
            created_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (date, stock)
        );
        """
        """

        CREATE TABLE IF NOT EXISTS param_model_predictions (
            date DATE NOT NULL,
            stock TEXT NOT NULL,
            sma_short INT,
            sma_long INT,
            rsi_thresh FLOAT,
            confidence FLOAT,
            expected_sharpe FLOAT,
            created_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (date, stock)
        );
        """
        """

        CREATE TABLE IF NOT EXISTS price_model_predictions (
            date DATE NOT NULL,
            stock TEXT NOT NULL,
            predicted_price FLOAT,
            prediction_horizon INT,
            model_version TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT now(),
            PRIMARY KEY (date, stock)
        );
        """
    }

    for table_name, create_sql in tables_to_create.items():
        already_exists = table_exists(table_name)
        execute_raw_sql(create_sql)
        if already_exists:
            print(f"✅ Table already exists: {table_name}")
        else:
            print(f"🆕 Table created: {table_name}")

    print("\n✅ PostgreSQL schema initialization complete.")

if __name__ == "__main__":
    init_postgres()
