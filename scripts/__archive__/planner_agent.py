from core.data_provider import load_data, save_data
from core.logger import logger
# agents/planner_agent.py
import os
import pandas as pd
from datetime import datetime
from agents.strategy_agent import StrategyAgent
from agents.execution_agent_sql import ExecutionAgent
from agents.memory_agent import MemoryAgent
from stock_selecter.auto_filter_selector import auto_select_filter
from services.feedback_loop import update_training_data
from models.stock_filter_predictor import run_stock_filter
import fundamentals.fundamental_data_extractor as fde
from config.paths import PATHS
from utils.file_io import save_dataframe

TOP_N = 5

class PlannerAgent:
    def __init__(self):
        self.strategy_agent = StrategyAgent()
        self.execution_agent = ExecutionAgent(dry_run=False)
        self.memory_agent = MemoryAgent()

    def run_weekly_routine(self, current_date=None):
        if current_date is None:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"📆 Current simulation date: {current_date}")

        logger.start("\n🧭 Starting PlannerAgent weekly routine...")

        logger.info("📊 Fetching fresh fundamentals...")
        fde.fetch_all()

        logger.info("🧠 Running ML-based stock filter...")
        run_stock_filter()

        if not os.path.exists(PATHS["ml_selected_stocks"]):
            logger.error("❌ Filtered stock list not found. Aborting.")
            return

        stocks = load_data("ml_selected_stocks" )["stock"].tolist()
        logger.info(f"🔍 {len(stocks)} stocks selected by ML filter.")

        results = []
        for stock in stocks:
            result = self.strategy_agent.evaluate(stock)
            if result:
                results.append(result)

        if not results:
            logger.warning("⚠️ No valid strategy results this week. Skipping trade execution.")
            return

        df = pd.DataFrame(results).sort_values(by="sharpe", ascending=False).head(TOP_N)
        save_data(df, "recommendations")  # SQLIFIED

        logger.success(f"✅ Saved top {TOP_N} weekly trades to {PATHS['recommendations']}")

        self.execution_agent.run()
        self.memory_agent.update()

        logger.info("🔁 Running feedback loop to update training data and retrain models...")
        update_training_data()

        logger.success("🏁 PlannerAgent routine complete.")

if __name__ == "__main__":
    PlannerAgent().run_weekly_routine()
