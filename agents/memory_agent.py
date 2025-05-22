# agents/memory_agent.py

from db.postgres_manager import run_query
from db.conflict_utils import insert_with_conflict_handling
from core.data_provider import load_data
from core.logger import logger
from core.time_context import get_simulation_date
from core.model_io import save_model
from models.train_exit_model import train_exit_model
from models.train_stock_filter_model import train_stock_filter_model
from models.train_dual_model_sql import train_dual_model
from models.meta_strategy_selector import train_meta_model
from core.config import settings
from services.feedback_loop import update_training_data
from rl.rl_finetune import finetune_rl  # 🆕 Add RL finetune hook

import pandas as pd


class MemoryAgent:
    def __init__(self):
        self.today = pd.to_datetime(get_simulation_date()).normalize()
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_date = self.today.date()

    def archive_table(self, logical_name: str):
        phys = settings.table_map[logical_name]
        archive = f"{phys}_archive_{self.today.strftime('%Y_%m_%d')}"
        try:
            logger.info(f"🗄️ Archiving {phys} → {archive}")
            run_query(
                f'CREATE TABLE IF NOT EXISTS "{archive}" AS SELECT * FROM "{phys}";',
                fetchall=False
            )
            logger.success(f"✅ Archived {phys}")
        except Exception as e:
            logger.error(f"❌ Archive failed for {phys}: {e}")

    def summarize_weekly_performance(self):
        df = load_data(settings.paper_trades_table)
        if df is None or df.empty:
            logger.info("📬 No paper trades to summarize.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        today_trades = df[df["timestamp"].dt.date == self.today_date]
        if today_trades.empty:
            logger.info("📬 No trades executed today.")
            return

        grouped = today_trades.sort_values("timestamp").groupby("stock")
        returns = []
        for stock, trades in grouped:
            buys = trades[trades["action"] == "buy"]["price"].values
            sells = trades[trades["action"] == "sell"]["price"].values
            pairs = zip(buys[:len(sells)], sells)
            for buy, sell in pairs:
                returns.append((sell - buy) / buy)

        if returns:
            avg_ret = sum(returns) / len(returns)
            logger.info(f"📊 Today's avg paper-trade return: {avg_ret:.2%} ({len(returns)} trades)")
        else:
            logger.info("📬 No valid buy-sell pairs for return calculation.")

    def check_retraining_needed(self):
        logger.start("🧠 Checking retraining thresholds...")
        try:
            pt_cnt = run_query(
                f"SELECT COUNT(*) FROM {settings.paper_trades_table}", fetchone=True
            )[0]
        except Exception:
            pt_cnt = 0

        try:
            td_cnt = run_query(
                f"SELECT COUNT(*) FROM {settings.training_data_table}", fetchone=True
            )[0]
        except Exception:
            td_cnt = 0

        logger.info(f"🧠 Paper trades={pt_cnt}, Training samples={td_cnt}")

        retrained = False
        if pt_cnt >= settings.retrain.paper_trades_threshold:
            logger.start("📚 Retraining exit model…")
            m = train_exit_model()
            save_model(settings.model_names["exit"], m)
            retrained = True

        if td_cnt >= settings.retrain.training_data_threshold:
            logger.start("📚 Retraining filter model…")
            m = train_stock_filter_model()
            save_model(settings.model_names["filter"], m)

            logger.start("📚 Retraining dual model…")
            m = train_dual_model()
            save_model(settings.model_names["dual"], m)

            logger.start("📚 Retraining meta model…")
            m = train_meta_model()
            save_model(settings.model_names["meta"], m)

            retrained = True

        if retrained:
            logger.success("✅ Retraining complete.")

        # ✅ Finetune RL based on replay buffer
        logger.start("🎯 Checking RL finetune opportunity...")
        try:
            finetune_rl(model_path="models/rl_policy.zip", steps=5000)
        except Exception as e:
            logger.warning(f"⚠️ RL finetune failed: {e}")

        if not retrained:
            logger.info("🔵 No retraining needed.")

    def feedback_loop(self):
        update_training_data()

    def update(self):
        logger.start("\n🚀 MemoryAgent full weekly update…")
        self.summarize_weekly_performance()
        self.feedback_loop()
        self.check_retraining_needed()
        if getattr(settings, "enable_archiving", False):
            for tbl in settings.archive_order:
                self.archive_table(tbl)
        logger.success("✅ MemoryAgent update done.")


if __name__ == "__main__":
    MemoryAgent().update()