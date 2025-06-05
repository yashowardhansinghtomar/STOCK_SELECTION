# agents/planner_agent_sql.py

from datetime import datetime
import random
import pandas as pd
from tqdm import tqdm
import pytz
import warnings
from sqlalchemy import text

from core.logger.logger import logger
from core.config.config import settings
from core.time_context.time_context import get_simulation_date
from core.data_provider.data_provider import fetch_stock_data, load_data, save_data
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.predict.predictor import predict_dual_model
from core.predict.predict_param_model import predict_param_config
from core.skiplist.skiplist import add_to_skiplist, is_in_skiplist
from core.logger.system_logger import log_event
from db.db import SessionLocal
from db.models import Base
from db.postgres_manager import run_query, get_all_symbols
from db.conflict_utils import insert_with_conflict_handling
from core.system_state import get_system_config

from agents.execution.execution_agent_sql import ExecutionAgentSQL
from agents.memory.memory_agent import MemoryAgent
from agents.risk_management_agent import RiskManagementAgent
from agents.strategy.rl_strategy_agent import RLStrategyAgent
from agents.arbitration.signal_arbitration_agent import SignalArbitrationAgent
from agents.strategy.strategy_agent import StrategyAgent
from stock_selecter.auto_filter_selector import auto_select_filter
import core.data_provider.fundamentals.fundamental_data_extractor as fde

warnings.filterwarnings("ignore", category=FutureWarning)
IST = pytz.timezone("Asia/Kolkata")
BAD_PATTERNS = ["NIFTY", "IDX", "SG", "COMMODITIES", "CONSUMPTION"]

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
        self.prefix = "🌅 [DAILY] "
        self.today = pd.to_datetime(get_simulation_date()).astimezone(IST).normalize()
        logger.info(f"{self.prefix}"+str(f"📆 SIMULATED_DATE initially set to: {self.today.date()}"))

        self.strategy_agent = StrategyAgent()
        self.execution_agent = ExecutionAgentSQL(self.session, dry_run=dry_run)
        self.memory_agent = MemoryAgent()
        self.signal_arbitrator = SignalArbitrationAgent()
        self.risk_agent = RiskManagementAgent()
        self.rl_agent = RLStrategyAgent()

        self.force_fetch = force_fetch
        self.force_enrich = force_enrich
        self.force_filter = force_filter
        self.force_eval = force_eval
        self.stock_whitelist = stock_whitelist

        self.top_n = settings.top_n
        self.max_eval = settings.max_eval
        self.skiplist_table = "skiplist_stocks"
        self.suppress_skiplist_logs = True

    def run(self):
        logger.start("Running PlannerAgentSQL...", prefix=self.prefix)
        log_event("PlannerAgentSQL", "run", "start", "running")
        try:
            self._fetch_fundamentals()
            self._fetch_price_history()
            self._refresh_features()
            self._filter_stocks()
            self._evaluate_stocks()
            self._execute_trades()
            self._update_systems()
            logger.success("\ud83c\udfcb\ufe0f PlannerAgentSQL routine complete.", prefix=self.prefix)
            log_event("PlannerAgentSQL", "run", "complete", "success")

        except Exception as e:
            logger.error(f"{self.prefix}"+str(f"\ud83d\udd25 Critical failure: {e}"))
            log_event("PlannerAgentSQL", "run", "complete", "failure", meta={"error": str(e)})
            raise
        finally:
            self.session.close()

    def _fetch_fundamentals(self):
        if not settings.use_fundamentals:
            logger.info(f"{self.prefix}"+str("🚫 Fundamentals disabled via config. Skipping fetch."))
            return

        fundamentals = load_data(settings.fundamentals_table)
        needs_fetch = (
            self.force_fetch
            or fundamentals is None
            or fundamentals.empty
            or pd.to_datetime(fundamentals["imported_at"].max()).date() != self.today.date()
        )
        if needs_fetch:
            logger.info(f"{self.prefix}"+str("📥 Fetching fresh fundamentals..."))
            fde.fetch_all()
            logger.success("✅ Fundamentals fetched successfully.", prefix=self.prefix)
        else:
            logger.success("📦 Fundamentals already available. Skipping fetch.", prefix=self.prefix)

    def _fetch_price_history(self):
        logger.info(f"{self.prefix}"+str("📥 Fetching price history for all symbols..."))
        inst = load_data(settings.instruments_table)
        if inst is None or inst.empty:
            logger.warnings("⚠️ No instruments found; skipping price fetch.", prefix=self.prefix)
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
                    logger.info(f"{self.prefix}"+str(f"⏭️ Skipping {sym} (in skiplist)"))
                continue
            try:
                df_new = fetch_stock_data(symbol=sym, end=self.today, interval=settings.price_fetch_interval, days=1)
                if df_new is None or df_new.empty:
                    logger.warnings(f"⚠️ No data for {sym}. Adding to skiplist.", prefix=self.prefix)
                    run_query(f"INSERT INTO {self.skiplist_table}(stock, reason) VALUES (%s, %s) ON CONFLICT DO NOTHING", params=(sym, "no_data"), fetchall=False)
                else:
                    fetched += 1
            except Exception as e:
                logger.warnings(f"⚠️ Fetch failed for {sym}: {e}. Adding to skiplist.", prefix=self.prefix)
                run_query(f"INSERT INTO {self.skiplist_table}(stock, reason) VALUES (%s, %s) ON CONFLICT DO NOTHING", params=(sym, str(e)), fetchall=False)

        logger.success(f"✅ Price fetch complete: {fetched} succeeded.", prefix=self.prefix)

    def _refresh_features(self):
        logger.info(f"{self.prefix}"+str("🔁 Generating multi-interval features..."))
        for stock in get_all_symbols():
            if is_in_skiplist(stock):
                continue
            if any(p in stock.upper() for p in BAD_PATTERNS):
                logger.warnings(f"⏩ Skipping {stock} due to bad pattern. Adding to skiplist.", prefix=self.prefix)
                add_to_skiplist(stock, reason="bad_pattern")
                continue
            enrich_multi_interval_features(stock=stock, sim_date=self.today, intervals=["day"])
        logger.success("✅ Feature refresh complete.", prefix=self.prefix)

    def _filter_stocks(self):
        feats = load_data(settings.feature_table)
        if feats is None or feats.empty:
            logger.error(f"{self.prefix}"+str("❌ No feature data available; aborting filter step."))
            return

        max_feat = pd.to_datetime(feats["date"], errors="coerce").max().normalize()
        expected = self.today - pd.offsets.BDay(1)

        if max_feat < expected:
            logger.error(f"{self.prefix}"+str(f"❌ Features outdated (max: {max_feat.date()} < expected: {expected.date()}); aborting."))
            return

        selected = load_data(settings.ml_selected_stocks_table)
        needs_filter = (
            self.force_filter
            or selected is None
            or selected.empty
            or pd.to_datetime(selected["imported_at"].max()).date() != self.today.date()
        )

        if needs_filter:
            try:
                logger.info(f"{self.prefix}"+str("🧠 Running ML-based stock filter..."))
                auto_select_filter()
                logger.success("✅ Stock filtering complete.", prefix=self.prefix)
            except RuntimeError as e:
                logger.warnings(f"⚠️ Filter failed: {e}", prefix=self.prefix)
                return
        else:
            logger.success("📦 Stocks already filtered for today.", prefix=self.prefix)

    def _evaluate_stocks(self):
        from models.joint_policy import JointPolicyModel
        model = JointPolicyModel.load()

        df_filtered = load_data(settings.ml_selected_stocks_table)
        if df_filtered is None or df_filtered.empty:
            logger.warnings("⚠️ No ML-selected stocks. Using fallback.", prefix=self.prefix)
            stocks = settings.fallback_stocks
        else:
            df_filtered = df_filtered[pd.to_datetime(df_filtered["imported_at"]).dt.date == self.today.date()]
            stocks = df_filtered["stock"].dropna().unique().tolist()
            logger.info(f"{self.prefix}📊 Stocks to evaluate: {len(stocks)}")

        if self.stock_whitelist:
            stocks = [s for s in stocks if s in self.stock_whitelist]
            logger.info(f"{self.prefix}📋 Whitelist applied: {len(stocks)}")

        random.shuffle(stocks)
        eval_limit = min(len(stocks), self.max_eval)
        logger.info(f"{self.prefix}🔍 Evaluating top {eval_limit} stocks...")


        config = get_system_config()
        policy_mode = config.get("policy_mode", "mix").lower()
        rl_allocation = int(config.get("rl_allocation", 10)) / 100.0


        all_signals = []
        for stock in tqdm(stocks[:eval_limit], desc="Evaluating"):
            signals = []

            use_rl = policy_mode == "rl"
            use_rf = policy_mode == "rf"
            mix_mode = policy_mode == "mix"

            if mix_mode:
                use_rl = random.uniform(0, 1) < rl_allocation
                use_rf = not use_rl

            if use_rl:
                rl_sig = self.rl_agent.evaluate(stock)
                if rl_sig:
                    signals.append(rl_sig)

            if use_rf:
                ml_sig = self.strategy_agent.evaluate(stock)
                if ml_sig:
                    signals.append(ml_sig)

            final_signal = self.signal_arbitrator.arbitrate(signals)
            if final_signal:
                # 👉 Joint policy inference
                try:
                    joint = model.predict([final_signal["features"]])[0]
                    final_signal["joint_policy_decision"] = {
                        "position_size": joint[0],
                        "exit_days": joint[1],
                    }
                except Exception as e:
                    logger.warning(f"[PLANNER] Joint model failed on {stock}: {e}")

                all_signals.append(final_signal)

        if all_signals:
            df = pd.DataFrame(all_signals).sort_values(by="confidence", ascending=False).head(self.top_n)
            insert_with_conflict_handling(df, settings.recommendations_table)
            logger.success(f"✅ Saved {len(df)} final signals.", prefix=self.prefix)
        else:
            logger.warnings("❌ No valid signals produced.", prefix=self.prefix)

    def _execute_trades(self):
        logger.info(f"{self.prefix}"+str("💼 Executing trades..."))
        self.execution_agent.run()

    def _update_systems(self):
        logger.info(f"{self.prefix}"+str("🧠 Updating memory and training systems..."))
        self.memory_agent.update()
        self.strategy_agent.log_summary()

if __name__ == "__main__":
    PlannerAgentSQL().run()
