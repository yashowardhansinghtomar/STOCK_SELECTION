# agents/execution_agent_sql.py

import os
import time
from datetime import datetime

import pandas as pd
from core.time_context import get_simulation_date
from core.config import settings
from core.data_provider import load_data, fetch_stock_data
from core.logger import logger
from db.conflict_utils import insert_with_conflict_handling
from db.postgres_manager import run_query
from services.exit_policy_evaluator import get_exit_probability  # ✅ Updated

CAPITAL_PER_TRADE = settings.capital_per_trade     # e.g. 10000
TABLE_TRADES      = settings.trades_table          # e.g. "trades"
TABLE_OPEN_POS    = settings.open_positions_table  # e.g. "open_positions"
TABLE_RECS        = settings.recommendations_table
TABLE_PARAM_PRED  = "param_model_predictions"
TABLE_FILTER_PRED = "filter_model_predictions"


def safe_load_table(name: str, cols: list) -> pd.DataFrame:
    df = load_data(name)
    if df is None:
        logger.warning(f"⚠️ Table '{name}' not found. Using empty DataFrame.")
        return pd.DataFrame(columns=cols)
    return df


class ExecutionAgentSQL:
    def __init__(self, session, dry_run: bool = False):
        self.session = session
        self.dev_mode = os.getenv("DEV_MODE") == "1"
        self.dry_run = dry_run
        self.today = pd.to_datetime(get_simulation_date())
        self.now_str = self.today.strftime("%Y-%m-%d %H:%M")
        self.today_str = self.today.strftime("%Y-%m-%d")

    def load_signals(self) -> pd.DataFrame:
        df = safe_load_table(TABLE_RECS, settings.recommendation_columns)

        if "trade_triggered" not in df.columns:
            logger.error("❌ Missing required column `trade_triggered` in recommendations; cannot filter signals.")
            return pd.DataFrame()

        df = df[df["trade_triggered"] == 1]
        df = df.rename(columns={"stock": "symbol"})

        if df.empty:
            logger.info("⚠️ No trade-triggered recommendations for today; proceeding with exits only.")
        else:
            logger.info(f"📈 Loaded {len(df)} trade signals for execution.")

        return df.head(settings.top_n) if self.dev_mode else df

    def load_open_positions(self) -> pd.DataFrame:
        df = safe_load_table(TABLE_OPEN_POS, ["stock", "entry_price", "entry_date", "strategy_config"])
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "stock"})
        if "stock" not in df.columns:
            logger.warning("⚠️ Open-positions table has no 'stock' column; returning empty positions.")
            return pd.DataFrame(columns=["stock", "entry_price", "entry_date", "strategy_config"])
        for col in ("entry_price", "entry_date", "strategy_config"):
            if col not in df.columns:
                df[col] = None
        return df[["stock", "entry_price", "entry_date", "strategy_config"]]

    def load_today_ohlc(self, symbol: str):
        df = fetch_stock_data(symbol, days=1)
        if df is None or df.empty:
            logger.error(f"❌ No OHLC data for {symbol}")
            return None
        row = df.iloc[-1]
        return row["open"], row["high"], row["low"], row["close"]

    def exit_trades(self, open_positions: pd.DataFrame):
        if open_positions.empty:
            return open_positions.copy(), pd.DataFrame(
                columns=[
                    "timestamp", "stock", "action", "price",
                    "strategy_config", "signal_reason", "source", "imported_at"
                ]
            )

        remaining, exited = [], []
        for _, pos in open_positions.iterrows():
            sym = pos["stock"]
            ohlc = self.load_today_ohlc(sym)
            if not ohlc:
                remaining.append(pos)
                continue

            proba = get_exit_probability(pos)

            if proba >= 0.6:
                logger.success(f"✅ Exiting {sym} with high confidence ({proba:.2f})")
                exited.append({
                    "timestamp":       self.now_str,
                    "stock":           sym,
                    "action":          "sell",
                    "price":           ohlc[-1],
                    "strategy_config": pos.get("strategy_config", ""),
                    "signal_reason":   "ml_exit_high_confidence",
                    "source":          "execution_agent",
                    "imported_at":     self.now_str,
                })
            elif proba >= 0.4:
                logger.info(f"⏳ Skipping borderline exit for {sym} (proba={proba:.2f})")
                remaining.append(pos)
            else:
                logger.info(f"🚫 Holding {sym}, exit proba too low ({proba:.2f})")
                remaining.append(pos)

        remaining_df = pd.DataFrame(remaining, columns=open_positions.columns)
        exited_df = pd.DataFrame(exited)
        return remaining_df, exited_df

    def enter_trades(self, signals: pd.DataFrame, open_positions: pd.DataFrame):
        new_positions, entry_logs, param_preds, filter_preds = [], [], [], []
        seen = set(open_positions["stock"].tolist())

        for _, sig in signals.iterrows():
            sym = sig["symbol"]
            if sym in seen:
                continue

            ohlc = self.load_today_ohlc(sym)
            if not ohlc:
                continue
            _, _, _, close = ohlc
            if close <= 0:
                continue

            qty = int(CAPITAL_PER_TRADE / close)
            if qty < 1:
                continue

            logger.success(f"✅ Entering {sym} at {close:.2f}")
            new_positions.append({
                "stock": sym,
                "entry_price": close,
                "entry_date":  self.now_str,
                "strategy_config": sig.get("strategy_config", ""),
                "interval": sig.get("interval", "day")  # 🆕
            })


            entry_logs.append({
                "timestamp":       self.now_str,
                "stock":           sym,
                "action":          "buy",
                "price":           close,
                "strategy_config": sig.get("strategy_config", ""),
                "signal_reason":   "signal_generated",
                "source":          sig.get("source", "execution_agent"),
                "imported_at":     self.now_str,
                "interval":        sig.get("interval", "day")  # 🆕 Add interval tag
            })

            param_preds.append({
                "date":            self.today.date(),
                "stock":           sym,
                "sma_short":       sig.get("sma_short"),
                "sma_long":        sig.get("sma_long"),
                "rsi_thresh":      sig.get("rsi_thresh"),
                "confidence":      sig.get("confidence"),
                "expected_sharpe": sig.get("sharpe"),
                "created_at":      self.today
            })
            if pd.notna(sig.get("confidence")):
                filter_preds.append({
                    "date":       self.today.date(),
                    "stock":      sym,
                    "score":      sig.get("confidence"),
                    "rank":       sig.get("rank"),
                    "confidence": sig.get("confidence"),
                    "decision":   "buy",
                    "created_at": self.today
                })

        if entry_logs and not self.dry_run:
            insert_with_conflict_handling(pd.DataFrame(entry_logs), TABLE_TRADES)
        if param_preds and not self.dry_run:
            df = pd.DataFrame(param_preds).copy()
            df = df.fillna({
                "sma_short":      0,
                "sma_long":       0,
                "rsi_thresh":     0,
                "confidence":     0.0,
                "expected_sharpe": 0.0,
            })
            df["sma_short"] = df["sma_short"].astype(int)
            df["sma_long"]  = df["sma_long"].astype(int)
            df["rsi_thresh"] = df["rsi_thresh"].astype(int)
            insert_with_conflict_handling(df, TABLE_PARAM_PRED)
        if filter_preds and not self.dry_run:
            insert_with_conflict_handling(pd.DataFrame(filter_preds), TABLE_FILTER_PRED)

        return pd.concat([open_positions, pd.DataFrame(new_positions)], ignore_index=True)

    def run(self):
        logger.start("\n🚀 ExecutionAgentSQL starting.")
        start_t = time.time()

        signals = self.load_signals()
        open_positions = self.load_open_positions()

        open_positions, exits = self.exit_trades(open_positions)
        if not exits.empty and not self.dry_run:
            insert_with_conflict_handling(exits, TABLE_TRADES)
            logger.info(f"🚪 Exited {len(exits)} positions.")

        open_positions = self.enter_trades(signals, open_positions)

        try:
            run_query(f'DELETE FROM "{TABLE_OPEN_POS}"', fetchall=False)
        except Exception as e:
            logger.warning(f"⚠️ Could not clear open positions: {e}")

        cols = ["stock", "entry_price", "entry_date"]
        if not open_positions.empty and not self.dry_run:
            insert_with_conflict_handling(open_positions[cols], TABLE_OPEN_POS)
            logger.info(f"🗂️ {len(open_positions)} open positions saved.")
        else:
            logger.info("🗂️ No open positions to save.")

        elapsed = time.time() - start_t
        logger.success(f"✅ ExecutionAgentSQL complete in {elapsed:.2f}s.")
