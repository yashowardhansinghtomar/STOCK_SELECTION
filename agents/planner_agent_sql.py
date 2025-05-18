# agents/planner_agent_sql.py

from core.logger import logger
from core.data_provider import load_data, fetch_stock_data
from core.time_context import get_simulation_date
from agents.strategy_agent import StrategyAgent
from agents.execution_agent_sql import ExecutionAgentSQL
from agents.memory_agent import MemoryAgent
from agents.signal_arbitration_agent import SignalArbitrationAgent
from agents.risk_management_agent import RiskManagementAgent
from stock_selecter.auto_filter_selector import auto_select_filter
import fundamentals.fundamental_data_extractor as fde
from db.conflict_utils import insert_with_conflict_handling
from db.postgres_manager import run_query
from sqlalchemy import text
from core.config import settings
from db.models import Base
from db.db import engine, SessionLocal
import numpy as np

import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class PlannerAgentSQL:
    def __init__(
        self,
        force_fetch=False,
        force_enrich=False,
        force_filter=False,
        force_eval=False,
        dry_run=False,
        stock_whitelist=None,
    ):
        self.session = SessionLocal()
        Base.metadata.create_all(bind=self.session.get_bind())

        self.today = get_simulation_date()
        feats = load_data(settings.feature_table)
        if feats is not None and not feats.empty:
            max_feat = pd.to_datetime(feats["date"]).max().date()
            # if simulation date is a weekend or ahead of latest features, align back
            if pd.Timestamp(self.today).dayofweek >= 5 or pd.Timestamp(self.today).date() > max_feat:
                logger.info(f"🔄 Aligning simulation date to last available features: {max_feat}")
                self.today = str(max_feat)

        self.strategy_agent = StrategyAgent()
        self.execution_agent = ExecutionAgentSQL(self.session, dry_run=dry_run)
        self.memory_agent = MemoryAgent()

        self.force_fetch = force_fetch
        self.force_enrich = force_enrich
        self.force_filter = force_filter
        self.force_eval = force_eval
        self.stock_whitelist = stock_whitelist

        self.top_n = settings.top_n
        self.max_eval = settings.max_eval
        self.skiplist_table = "skiplist_stocks"
        self.suppress_skiplist_logs = True  # suppress verbose skip logs
        self.signal_arbitrator = SignalArbitrationAgent()
        self.risk_agent = RiskManagementAgent()

        
    def run_weekly_routine(self):
        try:
            logger.info(f"🔄 Simulation date is {self.today}")
            logger.start("\n🧭 Starting PlannerAgentSQL (Flexible + Resume Mode)...")
            self._fetch_fundamentals()
            self._refresh_features()
            self._fetch_price_history()
            self._filter_stocks()
            self._evaluate_stocks()
            self._execute_trades()
            self._update_systems()
            logger.success("🏁 PlannerAgentSQL routine complete.")
        except Exception as e:
            logger.error(f"🔥 Critical failure: {e}")
            raise
        finally:
            self.session.close()

    def _fetch_fundamentals(self):
        if not settings.use_fundamentals:
            logger.info("🚫 Fundamentals disabled via config. Skipping fetch.")
            return

        fundamentals = load_data(settings.fundamentals_table)
        needs_fetch = (
            self.force_fetch
            or fundamentals is None
            or fundamentals.empty
            or pd.to_datetime(fundamentals["imported_at"].max()).date()
            != datetime.strptime(self.today, "%Y-%m-%d").date()
        )
        if needs_fetch:
            logger.info("📥 Fetching fresh fundamentals...")
            fde.fetch_all()
            logger.success("✅ Fundamentals fetched successfully.")
        else:
            logger.success("📦 Fundamentals already fetched for today. Skipping fetch.")

    def _fetch_price_history(self):
        if not self.force_fetch:
            hist = load_data(settings.price_history_table)
        if (
            hist is not None
            and not hist.empty
            and self.today in pd.to_datetime(hist["date"]).dt.date.unique()
        ):

                logger.success(f"📦 Price history already fetched for {self.today}. Skipping entire fetch loop.")
                return

        logger.info("📥 Fetching fresh price history for all instruments…")
        inst = load_data(settings.instruments_table)
        if inst is None or inst.empty:
            logger.warning("⚠️ No instruments found; skipping price fetch.")
            return

        symbols = inst.get("tradingsymbol", inst.get("symbol")).tolist()

        try:
            df_skip = load_data(self.skiplist_table)
            skipset = set(df_skip["stock"].tolist()) if df_skip is not None else set()
        except Exception:
            skipset = set()

        fetched = 0
        for sym in tqdm(symbols, desc="Fetching price data"):
            if sym in skipset:
                if not self.suppress_skiplist_logs:
                    logger.info(f"⏭️ Skipping {sym} (in skiplist)")
                continue
            try:
                df_new = fetch_stock_data(
                    symbol=sym,
                    end=self.today,
                    interval=settings.price_fetch_interval,
                    days=1
                )
                if df_new is None or (hasattr(df_new, "empty") and df_new.empty):
                    logger.warning(f"⚠️ No data returned for {sym}. Adding to skiplist.")
                    run_query(
                        f"""INSERT INTO {self.skiplist_table}(stock, reason)
                            VALUES (%s, %s)
                            ON CONFLICT(stock) DO NOTHING;""",
                        params=(sym, "no_data"),
                        fetchall=False
                    )
                else:
                    fetched += 1
            except Exception as e:
                logger.warning(f"⚠️ Fetch failed for {sym}: {e}. Adding to skiplist.")
                run_query(
                    f"""INSERT INTO {self.skiplist_table}(stock, reason)
                        VALUES (%s, %s)
                        ON CONFLICT(stock) DO NOTHING;""",
                    params=(sym, str(e)),
                    fetchall=False
                )

        logger.success(
            f"✅ Price fetch complete: {fetched} succeeded, "
            f"{len(symbols) - fetched - len(skipset)} newly skipped, "
            f"{len(skipset)} pre-skipped."
        )

    def _refresh_features(self):
        logger.info("🔄 Refreshing materialized view stock_features…")
        with engine.begin() as conn:
            conn.execute(text("REFRESH MATERIALIZED VIEW stock_features;"))
        logger.success("✅ stock_features refreshed.")

    def _filter_stocks(self):
        # Pre-flight: ensure the feature view is up-to-date
        feats = load_data(settings.feature_table)
        if feats is None or feats.empty:
            logger.error("❌ No feature data available; aborting filter step.")
            return

        # parse and find latest feature date
        max_feat = pd.to_datetime(feats["date"]).max()
        expected = pd.to_datetime(self.today) - pd.Timedelta(days=1)

        # strip any timezone info so both are tz-naive
        if hasattr(max_feat, "tzinfo") and max_feat.tzinfo is not None:
            max_feat = max_feat.tz_localize(None)
        if hasattr(expected, "tzinfo") and expected.tzinfo is not None:
            expected = expected.tz_localize(None)

        # now the comparison won't raise
        if max_feat < expected:
            logger.error(
                f"❌ Features not up-to-date (max: {max_feat.date()} < expected: {expected.date()}); aborting filter step."
            )
            return

        selected = load_data(settings.selected_table)
        needs_filter = (
            self.force_filter
            or selected is None
            or selected.empty
            or pd.to_datetime(selected["imported_at"].max()).date()
            != datetime.strptime(self.today, "%Y-%m-%d").date()
        )

        if needs_filter:
            # Run filter with graceful handling of too-few-stocks error
            try:
                logger.info("🧠 Running ML-based stock filter...")
                auto_select_filter()
                logger.success("✅ Stock filtering complete.")
            except RuntimeError as e:
                logger.warning(f"{e}  Falling back from filter step.")
                return
        else:
            logger.success("📦 ML-selected stocks already available. Skipping filtering.")

    def _evaluate_stocks(self):
        df_filtered = load_data(settings.selected_table)
        if df_filtered is None or df_filtered.empty:
            logger.warning("⚠️ No ML-selected stocks. Falling back to manual list...")
            stocks = settings.fallback_stocks
        else:
            df_filtered = df_filtered[
                pd.to_datetime(df_filtered["imported_at"]).dt.date == datetime.today().date()
            ]
            stocks = df_filtered["stock"].dropna().unique().tolist()
            logger.info(f"📊 ML-selected stocks to evaluate: {len(stocks)}")

        if self.stock_whitelist:
            stocks = [s for s in stocks if s in self.stock_whitelist]
            logger.info(f"📋 Whitelisted stock filter applied: {len(stocks)} remaining")

        random.shuffle(stocks)
        eval_limit = min(len(stocks), self.max_eval)
        logger.info(f"🔍 Preparing to evaluate up to {eval_limit} stocks...")

        results = []
        for stock in tqdm(stocks[:eval_limit], desc="Evaluating strategies"):
            res = self.strategy_agent.evaluate(stock)
            if res:
                results.append(res)

        if not results:
            logger.error("❌ No valid strategies evaluated. Skipping trade execution.")
            return

        df = pd.DataFrame(results).sort_values(by="sharpe", ascending=False).head(self.top_n)
        df_sql = df[settings.recommendation_columns]
        param_cols = ["stock", "sma_short", "sma_long", "rsi_thresh", "confidence", "sharpe"]
        df_sql_param = df[param_cols].copy()

        # Ensure no NaNs and cast types safely
        df_sql_param = df_sql_param.fillna(0)
        df_sql_param["date"] = get_simulation_date()

        df_sql_param["sma_short"] = df_sql_param["sma_short"].astype("int32", errors='ignore')
        df_sql_param["sma_long"] = df_sql_param["sma_long"].astype("int32", errors='ignore')
        df_sql_param["rsi_thresh"] = df_sql_param["rsi_thresh"].astype("int32", errors='ignore')
        df_sql_param["confidence"] = df_sql_param["confidence"].astype(float).fillna(0.0)
        df_sql_param["sharpe"] = df_sql_param["sharpe"].astype(float).fillna(0.0)

        # Replace nan/inf with 0 just in case
        df_sql_param = df_sql_param.replace([np.nan, np.inf, -np.inf], 0)

        insert_with_conflict_handling(df_sql, settings.recommendations_table)
        insert_with_conflict_handling(df_sql_param, "param_model_predictions")
        logger.success(f"✅ Top {self.top_n} strategy recommendations saved.")
       
        all_signals = []
        for stock in stocks[:eval_limit]:
            signals = [self.strategy_agent.evaluate(stock)]
            final_signal = self.signal_arbitrator.arbitrate(signals)
            if final_signal:
                all_signals.append(final_signal)
    
        df = pd.DataFrame(all_signals)


    def _execute_trades(self):
        logger.info("💼 Executing trades with risk controls...")
        self.execution_agent.run()
        open_positions = load_data(settings.open_positions_table)
        controlled_positions = self.risk_agent.apply_risk_controls(open_positions)
        save_data(controlled_positions, settings.open_positions_table, if_exists="replace")

    def _update_systems(self):
        logger.info("🧠 Updating memory agent and feedback loop...")
        self.memory_agent.update()
        self.strategy_agent.log_summary()


if __name__ == "__main__":
    PlannerAgentSQL(
        force_fetch=False,
        force_enrich=False,
        force_filter=False,
        force_eval=False,
        dry_run=False,
    ).run_weekly_routine()