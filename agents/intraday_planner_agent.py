# agents/strategy_agent.py
import random
import pandas as pd
from tqdm import tqdm

from core.logger import logger
from core.config import settings
from core.data_provider import load_data
from core.time_context import get_simulation_date
from core.model_io import insert_with_conflict_handling

from agents.execution_agent_sql import ExecutionAgentSQL
from agents.risk_management_agent import RiskManagementAgent
from agents.signal_arbitration_agent import SignalArbitrationAgent
from agents.rl_strategy_agent import RLStrategyAgent
from agents.strategy_agent import StrategyAgent


class IntradayPlannerAgent:
    def __init__(self, dry_run=False, stock_whitelist=None):
        self.today = pd.to_datetime(get_simulation_date()).normalize()
        self.execution_agent = ExecutionAgentSQL(None, dry_run=dry_run)
        self.signal_arbitrator = SignalArbitrationAgent()
        self.risk_agent = RiskManagementAgent()
        self.rl_agent = RLStrategyAgent(model_name="ppo_intraday", intervals=["15m", "60m"])
        self.strategy_agent = StrategyAgent()
        self.top_n = settings.top_n
        self.max_eval = settings.max_eval
        self.stock_whitelist = stock_whitelist

    def run(self):
        logger.info("🚀 Starting Intraday Planner Agent...")

        df_filtered = load_data(settings.selected_table)
        if df_filtered is None or df_filtered.empty:
            logger.warning("⚠️ No ML-selected stocks. Using fallback list...")
            stocks = settings.fallback_stocks
        else:
            df_filtered = df_filtered[
                pd.to_datetime(df_filtered["imported_at"]).dt.date == self.today.date()
            ]
            stocks = df_filtered["stock"].dropna().unique().tolist()

        if self.stock_whitelist:
            stocks = [s for s in stocks if s in self.stock_whitelist]

        random.shuffle(stocks)
        eval_limit = min(len(stocks), self.max_eval)
        logger.info(f"🔎 Evaluating {eval_limit} stocks (intraday)...")

        signals = []
        for stock in tqdm(stocks[:eval_limit], desc="Intraday RL + ML evaluation"):
            sigs = []

            rl_sig = self.rl_agent.evaluate(stock)
            if rl_sig:
                sigs.append(rl_sig)

            ml_sig = self.strategy_agent.evaluate(stock)
            if ml_sig:
                sigs.append(ml_sig)

            final = self.signal_arbitrator.arbitrate(sigs)
            if final:
                signals.append(final)

        if not signals:
            logger.warning("❌ No valid intraday signals found.")
            return

        final_signals = []
        for sig in signals:
            if self.risk_agent.approve(sig):
                final_signals.append(sig)

        if not final_signals:
            logger.warning("⚠️ No signals passed risk checks.")
            return

        df = pd.DataFrame(final_signals).sort_values(by="confidence", ascending=False).head(self.top_n)
        logger.success(f"✅ Selected top {len(df)} intraday trades.")

        insert_with_conflict_handling(df, settings.recommendations_table)
        self.execution_agent.enter_trades(df.to_dict("records"))


if __name__ == "__main__":
    agent = IntradayPlannerAgent(dry_run=True)
    agent.run()