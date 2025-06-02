# workers/missed_trade_logger.py
def log_missed_trades(date):
    traded_symbols = fetch_traded_symbols(date)
    all_candidates = fetch_all_planner_candidates(date)

    for symbol in all_candidates:
        if symbol not in traded_symbols:
            pnl = backtest_simple_strategy(symbol, date)
            publish_event("TRADE_CLOSE", {
                "symbol": symbol,
                "exit_price": None,
                "pnl": pnl,
                "virtual": True
            })
