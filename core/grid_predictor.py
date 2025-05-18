from datetime import datetime
from backtesting import Backtest
from core.data_provider import fetch_stock_data
from core.logger import logger
from core.time_context import get_simulation_date
from db.postgres_manager import run_query


def predict_grid_config(stock: str, top_n: int = 3) -> list:
    logger.info(f"🔍 Running grid prediction for {stock} using backtesting.py optimizer...")
    end_date = get_simulation_date()
    df = fetch_stock_data(stock, end=end_date)
    if df is None or df.empty:
        logger.warning(f"⚠️ No price history for {stock}. Aborting grid prediction.")
        return []

    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume',
        'volume_proxy': 'Volume'
    })

    rows = run_query("SELECT sma_short, sma_long, rsi_thresh FROM grid_params")
    sma_shorts = sorted({r[0] for r in rows})
    sma_longs = sorted({r[1] for r in rows})
    rsi_thres = sorted({r[2] for r in rows})
    if not (sma_shorts and sma_longs and rsi_thres):
        logger.warning("⚠️ grid_params table is empty or malformed.")
        return []

    try:
        from core.backtest_bt import SMA_RSI
    except ImportError as e:
        logger.error(f"Cannot load backtest strategy (SMA_RSI): {e}")
        return []

    bt = Backtest(df, SMA_RSI, cash=10_000, commission=0.002)
    stats = bt.optimize(
        sma_short=sma_shorts,
        sma_long=sma_longs,
        rsi_thresh=rsi_thres,
        maximize='Return [%]',
        constraint=lambda p: p.sma_short < p.sma_long,
        return_heatmap=False
    )

    strat = stats._strategy
    best_params = {
        'sma_short': strat.sma_short,
        'sma_long': strat.sma_long,
        'rsi_thresh': strat.rsi_thresh
    }
    pred_return = stats['Return [%]']

    return [{
        'stock': stock,
        'recommended_config': best_params,
        'predicted_return': float(pred_return),
        'trade_triggered': 1
    }]


def persist_grid_recommendations(stocks: list[str], top_n: int = 1):
    today = get_simulation_date()

    for stock in stocks:
        recs = predict_grid_config(stock, top_n=top_n)
        if not recs:
            logger.warning(f"No grid rec for {stock}; skipping persist.")
            continue

        rec = recs[0]
        cfg = rec["recommended_config"]

        sma_short = int(cfg["sma_short"])
        sma_long = int(cfg["sma_long"])
        rsi_thresh = int(cfg["rsi_thresh"])
        pred_return = float(rec["predicted_return"])
        triggered = int(rec["trade_triggered"])

        # → update recommendations table
        run_query(
            """
            INSERT INTO recommendations
              (stock, date, sma_short, sma_long, rsi_thresh,
               predicted_return, trade_triggered, source, imported_at)
            VALUES
              (:stock, :date, :sma_short, :sma_long, :rsi_thresh,
               :predicted_return, :trade_triggered, :source, :imported_at)
            ON CONFLICT (stock, date) DO UPDATE SET
              sma_short        = EXCLUDED.sma_short,
              sma_long         = EXCLUDED.sma_long,
              rsi_thresh       = EXCLUDED.rsi_thresh,
              predicted_return = EXCLUDED.predicted_return,
              trade_triggered  = EXCLUDED.trade_triggered,
              source           = EXCLUDED.source,
              imported_at      = EXCLUDED.imported_at;
            """,
            params={
                "stock": stock,
                "date": today,
                "sma_short": sma_short,
                "sma_long": sma_long,
                "rsi_thresh": rsi_thresh,
                "predicted_return": pred_return,
                "trade_triggered": triggered,
                "source": "grid_predictor",
                "imported_at": datetime.now()
            },
            fetchall=False
        )

        # → update param_model_predictions table
        run_query(
            """
            INSERT INTO param_model_predictions
              (date, stock, sma_short, sma_long, rsi_thresh,
               confidence, expected_sharpe, created_at)
            VALUES
              (:date, :stock, :sma_short, :sma_long, :rsi_thresh,
               :confidence, :expected_sharpe, :created_at)
            ON CONFLICT (date, stock) DO UPDATE SET
              sma_short       = EXCLUDED.sma_short,
              sma_long        = EXCLUDED.sma_long,
              rsi_thresh      = EXCLUDED.rsi_thresh,
              confidence      = EXCLUDED.confidence,
              expected_sharpe = EXCLUDED.expected_sharpe,
              created_at      = EXCLUDED.created_at;
            """,
            params={
                "date": today,
                "stock": stock,
                "sma_short": sma_short,
                "sma_long": sma_long,
                "rsi_thresh": rsi_thresh,
                "confidence": None,
                "expected_sharpe": pred_return,
                "created_at": datetime.now()
            },
            fetchall=False
        )

        logger.success(f"✅ Persisted grid rec for {stock}")


if __name__ == "__main__":
    import multiprocessing
    from backtesting import backtesting
    backtesting.Pool = multiprocessing.Pool
    symbols = ["RELIANCE", "TCS", "INFY"]
    persist_grid_recommendations(symbols, top_n=1)
