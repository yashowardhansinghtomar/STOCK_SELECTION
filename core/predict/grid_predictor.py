from datetime import datetime
import pandas as pd
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import fetch_stock_data
from core.config.strategy_config import StrategyConfig, ExitRule
from core.backtest_bt import run_backtest_config
from db.postgres_manager import run_query
import json


def predict_grid_config(stock: str, top_n: int = 3) -> list:
    logger.info(f"🔍 Running grid prediction for {stock} with entry+exit combos…")
    end_date = get_simulation_date()
    df = fetch_stock_data(stock, end=end_date)
    if df is None or df.empty:
        logger.warnings(f"⚠️ No price history for {stock}. Aborting grid prediction.")
        return []

    rows = run_query("SELECT sma_short, sma_long, rsi_thresh FROM grid_params")
    sma_shorts = sorted({r[0] for r in rows})
    sma_longs = sorted({r[1] for r in rows})
    rsi_thres = sorted({r[2] for r in rows})
    if not (sma_shorts and sma_longs and rsi_thres):
        logger.warnings("⚠️ grid_params table is empty or malformed.")
        return []

    # Define exit rules to try
    exit_rules = [
        ExitRule(kind="fixed_pct", stop_loss=0.03, take_profit=0.06, max_holding_days=10),
        ExitRule(kind="sma_cross", sma_window=20, max_holding_days=10),
        ExitRule(kind="time_stop", max_holding_days=5)
    ]

    results = []
    for s in sma_shorts:
        for l in sma_longs:
            if s >= l:
                continue
            for r in rsi_thres:
                for rule in exit_rules:
                    cfg = StrategyConfig(
                        sma_short=s,
                        sma_long=l,
                        rsi_entry=r,
                        exit_rule=rule
                    )
                    try:
                        metrics = run_backtest_config(stock, cfg, end=end_date)
                        if not metrics:
                            continue
                        results.append({
                            "stock": stock,
                            "recommended_config": cfg.dict(),
                            "predicted_return": metrics["total_return"],
                            "sharpe": metrics["sharpe"],
                            "max_drawdown": metrics["max_drawdown"],
                            "avg_trade_return": metrics["avg_trade_return"],
                            "trade_count": metrics["trade_count"],
                            "trade_triggered": 1
                        })
                    except Exception as e:
                        logger.warnings(f"❌ Backtest failed for {stock}: {e}")
                        continue

    if not results:
        return []

    df = pd.DataFrame(results)
    df = df.sort_values(by="sharpe", ascending=False).head(top_n)
    return df.to_dict(orient="records")


def persist_grid_recommendations(stocks: list[str], top_n: int = 1):
    today = get_simulation_date()

    for stock in stocks:
        recs = predict_grid_config(stock, top_n=top_n)
        if not recs:
            logger.warnings(f"No grid rec for {stock}; skipping persist.")
            continue

        for rec in recs:
            cfg = rec["recommended_config"]
            exit_rule = cfg.get("exit_rule", {})

            run_query(
                """
                INSERT INTO recommendations
                  (stock, date, sma_short, sma_long, rsi_thresh,
                   predicted_return, trade_triggered, source, imported_at, exit_rule)
                VALUES
                  (:stock, :date, :sma_short, :sma_long, :rsi_thresh,
                   :predicted_return, :trade_triggered, :source, :imported_at, :exit_rule)
                ON CONFLICT (stock, date) DO UPDATE SET
                  sma_short        = EXCLUDED.sma_short,
                  sma_long         = EXCLUDED.sma_long,
                  rsi_thresh       = EXCLUDED.rsi_thresh,
                  predicted_return = EXCLUDED.predicted_return,
                  trade_triggered  = EXCLUDED.trade_triggered,
                  source           = EXCLUDED.source,
                  imported_at      = EXCLUDED.imported_at,
                  exit_rule        = EXCLUDED.exit_rule;
                """,
                params={
                    "stock": stock,
                    "date": today,
                    "sma_short": int(cfg["sma_short"]),
                    "sma_long": int(cfg["sma_long"]),
                    "rsi_thresh": float(cfg["rsi_entry"]),
                    "predicted_return": float(rec["predicted_return"]),
                    "trade_triggered": int(rec["trade_triggered"]),
                    "source": "grid_predictor",
                    "imported_at": datetime.now(),
                    "exit_rule": json.dumps(exit_rule)
                },
                fetchall=False
            )

            logger.success(f"✅ Grid rec persisted for {stock} (Sharpe: {rec['sharpe']:.2f})")


if __name__ == "__main__":
    from backtesting import backtesting
    import multiprocessing

    backtesting.Pool = multiprocessing.Pool
    symbols = ["RELIANCE", "TCS", "INFY"]
    persist_grid_recommendations(symbols, top_n=1)
