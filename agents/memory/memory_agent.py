# agents/memory_agent.py

import pandas as pd
from core.config.config import settings
from core.data_provider.data_provider import load_data
from core.logger.logger import logger
from core.logger.system_logger import log_event
from core.model_io import save_model
from core.time_context.time_context import get_simulation_date
from db.conflict_utils import insert_with_conflict_handling
from db.postgres_manager import run_query
from db.replay_buffer_sql import count_by_stock

from models.meta_strategy_selector import train_meta_model
from models.train_dual_model_sql import train_dual_model
from models.train_exit_model import train_exit_model
from models.train_stock_filter_model import train_stock_filter_model
from rl.rl_finetune import finetune_rl
from agents.memory.feedback_loop import update_training_data


def top_stocks_with_replay_data(min_episodes: int = 30):
    try:
        buffer_counts = count_by_stock()
        return [
            (row["stock"], row["interval"])
            for row in buffer_counts
            if row.get("count", 0) >= min_episodes
        ]
    except Exception as e:
        logger.warnings(f"🧠 [MEMORY] ⚠️ Failed to fetch replay buffer counts: {e}")
        return []


class MemoryAgent:
    def __init__(self):
        self.today = pd.to_datetime(get_simulation_date()).tz_localize(None).normalize()
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_date = self.today.date()
        self.prefix = "🧠 [MEMORY] "

    def archive_table(self, logical_name: str):
        phys = settings.table_map[logical_name]
        archive = f"{phys}_archive_{self.today.strftime('%Y_%m_%d')}"
        try:
            logger.info(f"{self.prefix}🗄️ Archiving {phys} → {archive}")
            run_query(
                f'CREATE TABLE IF NOT EXISTS "{archive}" AS SELECT * FROM "{phys}";',
                fetchall=False
            )
            logger.success(f"{self.prefix}✅ Archived {phys}")
        except Exception as e:
            logger.error(f"{self.prefix}❌ Archive failed for {phys}: {e}")

    def summarize_weekly_performance(self):
        df = load_data(settings.tables.trades)
        if df is None or df.empty:
            logger.info(f"{self.prefix}📬 No paper trades to summarize.")
            return

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        today_trades = df[df["timestamp"].dt.date == self.today_date]
        if today_trades.empty:
            logger.info(f"{self.prefix}📬 No trades executed today.")
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
            logger.info(f"{self.prefix}📊 Today's avg paper-trade return: {avg_ret:.2%} ({len(returns)} trades)")
        else:
            logger.info(f"{self.prefix}📬 No valid buy-sell pairs for return calculation.")

    def check_retraining_needed(self):
        logger.start("🧠 Checking retraining thresholds...", prefix=self.prefix)

        trades = load_data(settings.tables.trades)
        training_data = load_data(settings.tables.training_data)
        if training_data is None:
            training_data = pd.DataFrame()

        recent_trades = pd.DataFrame()
        if trades is not None and not trades.empty and "run_timestamp" in trades.columns:
            trades["run_timestamp"] = pd.to_datetime(trades["run_timestamp"], errors="coerce")
            recent_trades = trades[trades["run_timestamp"] >= self.today - pd.Timedelta(weeks=4)]

        recent_td = pd.DataFrame()
        if not training_data.empty and "entry_date" in training_data.columns:
            training_data["entry_date"] = pd.to_datetime(training_data["entry_date"], errors="coerce")
            recent_td = training_data[training_data["entry_date"] >= self.today - pd.Timedelta(weeks=4)]

        avg_trades_per_week = len(recent_trades) / 4 if not recent_trades.empty else 0

        if avg_trades_per_week >= 30:
            trade_thresh = 100
            td_thresh = 1000
        elif avg_trades_per_week >= 15:
            trade_thresh = 60
            td_thresh = 600
        else:
            trade_thresh = 30
            td_thresh = 300

        logger.info(
            f"{self.prefix}📊 Rolling 4-week trade rate = {avg_trades_per_week:.1f}/week → "
            f"Thresholds: {trade_thresh} trades, {td_thresh} training rows"
        )

        retrained = False

        if len(recent_trades) >= trade_thresh:
            logger.start("📚 Retraining exit model…", prefix=self.prefix)
            m = train_exit_model()
            save_model(settings.model_names["exit"], m)
            retrained = True

        if len(recent_td) >= td_thresh:
            logger.start("📚 Retraining filter model…", prefix=self.prefix)
            m = train_stock_filter_model()
            save_model(settings.model_names["filter"], m)

            logger.start("📚 Retraining dual model…", prefix=self.prefix)
            m = train_dual_model()
            save_model(settings.model_names["dual"], m)

            logger.start("📚 Retraining meta model…", prefix=self.prefix)
            m = train_meta_model()
            save_model(settings.model_names["meta"], m)

            retrained = True

        logger.start("🎯 Checking RL finetune opportunity...", prefix=self.prefix)
        try:
            eligible = sorted(top_stocks_with_replay_data())
            logger.info(f"{self.prefix}🎯 Eligible stocks for RL finetune: {len(eligible)}")
            if not eligible:
                logger.info(f"{self.prefix}🟡 No stocks eligible for RL finetune.")
            for stock, interval in eligible:
                model_path = f"ppo_{stock}_{interval}"
                logger.info(f"{self.prefix}🔁 Finetuning RL model: {model_path}")
                finetune_rl(model_path=model_path, stock=stock, interval=interval, steps=5000)
        except Exception as e:
            logger.warnings(f"{self.prefix}⚠️ RL finetune failed: {e}")

        if not retrained:
            logger.info(f"{self.prefix}🔵 No retraining needed.")

    def feedback_loop(self):
        update_training_data()

    def update(self):
        logger.start("\n🚀 MemoryAgent full weekly update…", prefix=self.prefix)
        log_event("MemoryAgent", "update", "start", "running")
        self.summarize_weekly_performance()
        self.feedback_loop()
        self.check_retraining_needed()

        if getattr(settings, "enable_archiving", False):
            for tbl in settings.archive_order:
                self.archive_table(tbl)

        logger.success(f"{self.prefix}✅ MemoryAgent update done.")


if __name__ == "__main__":
    MemoryAgent().update()
