import json
import os
import time
from datetime import datetime

import pandas as pd
from core.time_context.time_context import get_simulation_date
from core.config.config import settings, FeatureGroupConfig
from core.data_provider.data_provider import load_data, fetch_stock_data
from core.logger.logger import logger
from core.logger.system_logger import log_event
from db.conflict_utils import insert_with_conflict_handling
from db.postgres_manager import run_query
from services.exit_policy_evaluator import get_exit_probability
from db.replay_buffer_sql import SQLReplayBuffer  as ReplayBuffer
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from db.replay_buffer_sql import insert_replay_episode
from core.event_bus import publish_event
from agents.execution.trade_execution_helper import TradeExecutionHelper

CAPITAL_PER_TRADE = settings.capital_per_trade
TABLE_TRADES = settings.tables.trades
TABLE_OPEN_POS = settings.tables.open_positions
TABLE_RECS = settings.tables.recommendations
TABLE_PARAM_PRED = settings.tables.predictions["param"]
TABLE_FILTER_PRED = settings.tables.predictions["filter"]

REPLAY_DIR = "ppo_buffers"
os.makedirs(REPLAY_DIR, exist_ok=True)

def safe_load_table(name: str, cols: list) -> pd.DataFrame:
    df = load_data(name)
    if df is None:
        logger.warnings(f"Table '{name}' not found. Using empty DataFrame.")
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
        self.prefix = "🔨 [EXEC] "
        self.executor = TradeExecutionHelper(self.today, dry_run=self.dry_run, prefix=self.prefix)

    def load_signals(self) -> pd.DataFrame:
        df = safe_load_table(TABLE_RECS, FeatureGroupConfig.recommendation_columns)
        if "trade_triggered" not in df.columns:
            logger.error(f"{self.prefix}"+str("Missing `trade_triggered` in recommendations."))
            return pd.DataFrame()
        df = df[df["trade_triggered"] == 1]
        df = df.rename(columns={"stock": "symbol"})
        if df.empty:
            logger.info(f"{self.prefix}"+str("No trade-triggered recommendations for today."))
        else:
            logger.info(f"{self.prefix}"+str(f"Loaded {len(df)} trade signals."))
        return df.head(settings.top_n) if self.dev_mode else df

    def load_open_positions(self) -> pd.DataFrame:
        df = safe_load_table(TABLE_OPEN_POS, [
            "stock", "entry_price", "entry_date",
            "sma_short", "sma_long", "rsi_thresh",
            "strategy_config", "interval"
        ])

        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "stock"})
        if "stock" not in df.columns:
            logger.warnings("Open-positions table missing 'stock'.", prefix=self.prefix)
            return pd.DataFrame(columns=["stock", "entry_price", "entry_date", "strategy_config"])
        for col in ("entry_price", "entry_date", "strategy_config"):
            if col not in df.columns:
                df[col] = None
        return df[["stock", "entry_price", "entry_date", "strategy_config"]]

    def load_today_ohlc(self, symbol: str):
        df = fetch_stock_data(symbol, days=1)
        if df is None or df.empty:
            logger.error(f"{self.prefix}"+str(f"No OHLC data for {symbol}"))
            return None
        row = df.iloc[-1]
        publish_event("BAR", {
            "symbol": symbol,
            "interval": "day",
            "ohlcv": row[["open", "high", "low", "close", "volume"]].to_dict(),
            "timestamp": self.now_str
        })
        return row["open"], row["high"], row["low"], row["close"]

    def exit_trades(self, open_positions: pd.DataFrame):
        if open_positions.empty:
            return open_positions.copy(), pd.DataFrame(columns=["timestamp", "stock", "action", "price", "strategy_config", "signal_reason", "source", "imported_at"])

        remaining, exited = [], []
        for _, pos in open_positions.iterrows():
            sym = pos["stock"]
            ohlc = self.load_today_ohlc(sym)
            if not ohlc:
                remaining.append(pos)
                continue
            proba = get_exit_probability(pos)
            if proba >= 0.6:
                logger.success(f"Exiting {sym} (proba={proba:.2f})", prefix=self.prefix)
                pnl = ohlc[-1] - pos["entry_price"]
                publish_event("TRADE_CLOSE", {
                    "symbol": sym,
                    "exit_price": ohlc[-1],
                    "reward": pnl,
                    "timestamp": self.now_str,
                    "strategy_config": pos.get("strategy_config", {})
                })
                exited.append({
                    "timestamp": self.now_str,
                    "stock": sym,
                    "action": "sell",
                    "price": ohlc[-1],
                    "strategy_config": pos.get("strategy_config", {}),
                    "signal_reason": "ml_exit_high_confidence",
                    "source": "execution_agent",
                    "imported_at": self.now_str,
                })
            elif proba >= 0.4:
                logger.info(f"{self.prefix}"+str(f"Borderline exit skipped for {sym} (proba={proba:.2f})"))
                remaining.append(pos)
            else:
                logger.info(f"Holding {sym}, exit proba too low ({proba:.2f})", prefix=self.prefix)
                # Estimate days held and unrealized PnL
                entry_price = float(pos.get("entry_price", 0))
                days_held = (self.today - pd.to_datetime(pos.get("entry_date", self.today))).days
                pnl = ohlc[-1] - entry_price if entry_price else 0.0
                cap_eff = pnl / entry_price if entry_price else 0.0
                self.publish_m2m_update(sym, days_held=days_held, capital_efficiency=cap_eff, unrealized_pnl=pnl)
                remaining.append(pos)

        remaining_df = pd.DataFrame(remaining, columns=open_positions.columns)
        exited_df = pd.DataFrame(exited)
        return remaining_df, exited_df

    def enter_trades(self, signals: pd.DataFrame, open_positions: pd.DataFrame):
        # Cap RL trades to allocation limit
        rl_alloc = settings.rl_allocation or 10
        max_rl_trades = int(len(signals) * rl_alloc / 100)

        if "source" not in signals.columns:
            signals["source"] = "unknown"

        rl_signals = signals[signals["source"] == "RL"]
        non_rl_signals = signals[signals["source"] != "RL"]

        # Apply cap and recombine
        signals = pd.concat([
            rl_signals.head(max_rl_trades),
            non_rl_signals
        ], ignore_index=True)

        new_positions, entry_logs, param_preds, filter_preds = [], [], [], []
        seen = set(open_positions["stock"].tolist())

        for _, sig in signals.iterrows():
            sym = sig["symbol"]
            if sym in seen:
                logger.info(f"{self.prefix}Already in position: {sym}")
                continue

            ohlc = self.load_today_ohlc(sym)
            if not ohlc:
                logger.warnings(f"Skipping {sym}: OHLC missing.", prefix=self.prefix)
                continue

            _, _, _, close = ohlc
            if close <= 0:
                logger.warnings(f"Non-positive price for {sym}: {close}", prefix=self.prefix)
                continue

            qty = int(CAPITAL_PER_TRADE / close)
            if qty < 1:
                logger.warnings(f"Qty < 1 for {sym} @ {close}.", prefix=self.prefix)
                continue

            interval = sig.get("interval", "day")
            strategy_cfg = sig.get("strategy_config", {})
            if isinstance(strategy_cfg, str):
                try:
                    strategy_cfg = json.loads(strategy_cfg)
                except Exception:
                    strategy_cfg = {}

            logger.success(f"Entering {sym} @ ₹{close:.2f}", prefix=self.prefix)

            publish_event("TRADE_OPEN", {
                "symbol": sym,
                "qty": qty,
                "price": close,
                "interval": interval,
                "strategy_config": strategy_cfg,
                "timestamp": self.now_str
            })

            new_positions.append({
                "stock": sym,
                "entry_price": close,
                "entry_date": self.now_str,
                "strategy_config": strategy_cfg,
                "interval": interval
            })

            entry_logs.append({
                "timestamp": self.now_str,
                "stock": sym,
                "action": "buy",
                "price": close,
                "strategy_config": strategy_cfg,
                "signal_reason": "signal_generated",
                "source": sig.get("source", "execution_agent"),
                "imported_at": self.now_str,
                "interval": interval
            })

            param_preds.append({
                "date": self.today.date(),
                "stock": sym,
                "sma_short": sig.get("sma_short"),
                "sma_long": sig.get("sma_long"),
                "rsi_thresh": sig.get("rsi_thresh"),
                "confidence": sig.get("confidence"),
                "expected_sharpe": sig.get("sharpe"),
                "created_at": self.today
            })

            if pd.notna(sig.get("confidence")):
                filter_preds.append({
                    "date": self.today.date(),
                    "stock": sym,
                    "score": sig.get("confidence"),
                    "rank": sig.get("rank"),
                    "confidence": sig.get("confidence"),
                    "decision": "buy",
                    "created_at": self.today
                })

            try:
                from bootstrap.trade_generator import Trade
                trade_obj = Trade(
                    symbol=sym,
                    timestamp=self.today,
                    order_type="MARKET",
                    price=close,
                    size=qty,
                    direction=1,
                    holding_period=pd.Timedelta(days=1),
                    exit_strategy="TIME_BASED",
                    meta={"exploration_type": sig.get("source", "unknown")}
                )

                result = self.executor.execute(
                    symbol=sym,
                    price=close,
                    size=qty,
                    strategy_config=strategy_cfg,
                    interval=interval,
                    trade_obj=trade_obj
                )

                if result:
                    self.executor.log_to_replay(result, strategy_cfg, interval)

                    # ✅ Also log to SQL replay buffer
                    try:
                        ReplayBuffer().add(result)

                    except Exception as e:
                        logger.warning(f"{self.prefix} Replay SQL insert failed for {sym}: {e}")

            except Exception as e:
                logger.warnings(f"{self.prefix}Execution or replay log failed for {sym}: {e}")

        if entry_logs and not self.dry_run:
            insert_with_conflict_handling(pd.DataFrame(entry_logs), TABLE_TRADES)

        if param_preds and not self.dry_run:
            df = pd.DataFrame(param_preds).copy().fillna({
                "sma_short": 0, "sma_long": 0, "rsi_thresh": 0,
                "confidence": 0.0, "expected_sharpe": 0.0
            })
            df["sma_short"] = df["sma_short"].astype(int)
            df["sma_long"] = df["sma_long"].astype(int)
            df["rsi_thresh"] = df["rsi_thresh"].astype(int)
            insert_with_conflict_handling(df, TABLE_PARAM_PRED)

        if filter_preds and not self.dry_run:
            insert_with_conflict_handling(pd.DataFrame(filter_preds), TABLE_FILTER_PRED)

        return pd.concat([open_positions, pd.DataFrame(new_positions)], ignore_index=True)

    def publish_m2m_update(self, symbol: str, days_held: int, capital_efficiency: float = 0.0, unrealized_pnl: float = 0.0):
        try:
            publish_event("M2M_PNL", {
                "event_type": "M2M_PNL",
                "timestamp": self.now_str,
                "symbol": symbol,
                "days_held": days_held,
                "capital_efficiency": capital_efficiency,
                "unrealized_pnl": unrealized_pnl,
                "interval": "day",
                "regime_tag": None  # optional if you calculate it
            })
            logger.debug(f"{self.prefix} M2M_PNL emitted for {symbol}")
        except Exception as e:
            logger.warning(f"{self.prefix} Failed to emit M2M_PNL for {symbol}: {e}")


    def run(self):
        logger.start("ExecutionAgentSQL starting.", prefix=self.prefix)
        log_event("ExecutionAgentSQL", "run", "start", "running")
        start_t = time.time()
        signals = self.load_signals()
        open_positions = self.load_open_positions()
        open_positions, exits = self.exit_trades(open_positions)
        if not exits.empty and not self.dry_run:
            insert_with_conflict_handling(exits, TABLE_TRADES)
            logger.info(f"{self.prefix}"+str(f"Exited {len(exits)} positions."))
        open_positions = self.enter_trades(signals, open_positions)
        try:
            run_query(f'DELETE FROM "{TABLE_OPEN_POS}"', fetchall=False)
        except Exception as e:
            logger.warning(f"Could not clear open positions: {e}", prefix=self.prefix)
        cols = ["stock", "entry_price", "entry_date", "sma_short", "sma_long", "rsi_thresh", "strategy_config", "interval"]
        if not open_positions.empty and not self.dry_run:
            insert_with_conflict_handling(open_positions[cols], TABLE_OPEN_POS)
            logger.info(f"{self.prefix}"+str(f"{len(open_positions)} open positions saved."))
        else:
            logger.info(f"{self.prefix}"+str("No open positions to save."))
        elapsed = time.time() - start_t
        logger.success(f"ExecutionAgentSQL complete in {elapsed:.2f}s.", prefix=self.prefix)
        log_event("ExecutionAgentSQL", "run", "complete", "success", meta={"elapsed_sec": round(elapsed, 2)})
