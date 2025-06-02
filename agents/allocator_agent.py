# core/agents/allocator_agent.py

from core.logger.logger import logger
from db.postgres_manager import run_query
from core.system_state import get_system_config, update_system_config

class AllocatorAgent:
    def __init__(self, lookback_days: int = 7):
        self.lookback_days = lookback_days

    def get_sharpe(self, model_type: str) -> float:
        query = f"""
        SELECT date, SUM(pnl) AS daily_pnl
        FROM paper_trades
        WHERE model_type = '{model_type}' AND date >= CURRENT_DATE - INTERVAL '{self.lookback_days} days'
        GROUP BY date
        ORDER BY date;
        """
        df = run_query(query)
        if df.empty or df['daily_pnl'].nunique() <= 1:
            return 0.0
        returns = df['daily_pnl'].pct_change().dropna()
        return returns.mean() / returns.std()

    def get_current_allocation(self) -> int:
        config = get_system_config()
        return int(config.get("rl_allocation", 10))

    def set_current_allocation(self, percent: int):
        update_system_config({"rl_allocation": percent})
        logger.info(f"[ALLOCATOR] RL allocation set to {percent}%")

    def run(self):
        sharpe_rl = self.get_sharpe('RL')
        sharpe_rf = self.get_sharpe('RF')
        logger.info(f"[ALLOCATOR] RL Sharpe: {sharpe_rl:.3f} vs RF: {sharpe_rf:.3f}")

        delta = sharpe_rl - sharpe_rf
        current = self.get_current_allocation()
        new_alloc = current

        if delta > -0.05:
            new_alloc = min(100, current + 10)
            reason = "RL performing better or equal"
        elif delta < -0.15:
            new_alloc = max(0, current - 10)
            reason = "RL significantly underperforming"
        else:
            reason = "RL slightly underperforming — holding allocation"

        if new_alloc != current:
            self.set_current_allocation(new_alloc)
            logger.info(f"[ALLOCATOR] RL allocation changed from {current}% to {new_alloc}% ({reason})")
        else:
            logger.info(f"[ALLOCATOR] RL allocation unchanged at {current}% ({reason})")
