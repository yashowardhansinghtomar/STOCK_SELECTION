# agents/intraday_planner_agent.py

import random
import pandas as pd
from tqdm import tqdm
import redis
import time
import os
from core.logger.logger import logger
from core.config.config import settings
from core.data_provider.data_provider import load_data
from core.time_context.time_context import get_simulation_date
from core.model_io import insert_with_conflict_handling

from agents.execution.execution_agent_sql import ExecutionAgentSQL
from agents.risk_management_agent import RiskManagementAgent
from agents.arbitration.signal_arbitration_agent import SignalArbitrationAgent
from agents.strategy.rl_strategy_agent import RLStrategyAgent
from agents.strategy.strategy_agent import StrategyAgent


class IntradayPlannerAgent:
    def __init__(self, dry_run=False, stock_whitelist=None):
        self.today = pd.to_datetime(get_simulation_date()).normalize()
        self.execution_agent = ExecutionAgentSQL(None, dry_run=dry_run)
        self.signal_arbitrator = SignalArbitrationAgent()
        self.risk_agent = RiskManagementAgent()
        self.rl_agent = RLStrategyAgent(model_name="ppo_intraday", intervals=["15minute", "60minute"])
        self.strategy_agent = StrategyAgent()
        self.top_n = settings.top_n
        self.max_eval = settings.max_eval
        self.stock_whitelist = stock_whitelist
        
        self.prefix = "📡 [INTRADAY] "
        self.redis = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=int(os.getenv("REDIS_PORT", "6379")), db=0)
        self.event_key = "feature_ready_1m"
        self.poll_interval = 300  # fallback polling every 5 mins

    def _fetch_updated_symbols(self) -> list:
        symbols = []
        while True:
            sym = self.redis.rpop(self.event_key)
            if not sym:
                break
            symbols.append(sym.decode())
        return list(set(symbols))  # deduplicate

    def _fallback_poll(self) -> list:
        df_filtered = load_data(settings.ml_selected_stocks_table)
        if df_filtered is None or df_filtered.empty:
            logger.warnings("⚠️ No ML-selected stocks in fallback.")
            return settings.fallback_stocks

        df_filtered = df_filtered[
            pd.to_datetime(df_filtered["imported_at"]).dt.date == self.today.date()
        ]
        return df_filtered["stock"].dropna().unique().tolist()

    def _process_symbols(self, updated_symbols):
        logger.info("🚀 Starting Intraday Planner Agent for updated symbols...")
        if self.stock_whitelist:
            updated_symbols = [s for s in updated_symbols if s in self.stock_whitelist]

        random.shuffle(updated_symbols)
        eval_limit = min(len(updated_symbols), self.max_eval)
        logger.info(f"🔎 Evaluating {eval_limit} stocks (intraday)...")

        policy_mode = getattr(settings, "policy_mode", "mix").lower()
        rl_allocation = getattr(settings, "rl_allocation", 10) / 100.0

        signals = []
        for stock in tqdm(updated_symbols[:eval_limit], desc="Intraday RL + ML evaluation"):
            sigs = []

            use_rl = policy_mode == "rl"
            use_rf = policy_mode == "rf"
            mix_mode = policy_mode == "mix"

            if mix_mode:
                use_rl = random.uniform(0, 1) < rl_allocation
                use_rf = not use_rl

            if use_rl:
                rl_sig = self.rl_agent.evaluate(stock)
                if rl_sig:
                    sigs.append(rl_sig)

            if use_rf:
                ml_sig = self.strategy_agent.evaluate(stock)
                if ml_sig:
                    sigs.append(ml_sig)

            final = self.signal_arbitrator.arbitrate(sigs)
            if final:
                signals.append(final)

        if not signals:
            logger.warnings("❌ No valid intraday signals found.")
            return

        final_signals = []
        for sig in signals:
            if self.risk_agent.approve(sig):
                final_signals.append(sig)

        if not final_signals:
            logger.warnings("⚠️ No signals passed risk checks.")
            return

        df = pd.DataFrame(final_signals).sort_values(by="confidence", ascending=False).head(self.top_n)
        logger.success(f"✅ Selected top {len(df)} intraday trades.")

        insert_with_conflict_handling(df, settings.tables.recommendations)

        # 🛠️ NEW: Exit logic for intraday trades before entering
        open_positions = load_data(settings.tables.open_positions)
        remaining, exited = self.execution_agent.exit_trades(open_positions)
        if not exited.empty:
            insert_with_conflict_handling(exited, settings.tables.trades)
            logger.success(f"✅ Exited {len(exited)} intraday positions.")

        # ✅ Enter new intraday positions
        self.execution_agent.enter_trades(df.to_dict("records"))

    def run(self):
        updated_symbols = self._fetch_updated_symbols()
        if not updated_symbols:
            logger.info("📭 No updated symbols for intraday eval right now.")
            return
        self._process_symbols(updated_symbols)

    def run_forever(self):
        logger.info("🌀 Intraday agent starting real-time loop...")
        last_poll = time.time()

        while True:
            updated = self._fetch_updated_symbols()
            if updated:
                self._process_symbols(updated)
            elif time.time() - last_poll > self.poll_interval:
                logger.info("⏰ Fallback polling triggered...")
                all_symbols = self._fallback_poll()
                self._process_symbols(all_symbols)
                last_poll = time.time()

            time.sleep(5)  # wait briefly between checks

if __name__ == "__main__":
    agent = IntradayPlannerAgent(dry_run=True)
    agent.run()
    # For real-time mode:
    # IntradayPlannerAgent(dry_run=False).run_forever()