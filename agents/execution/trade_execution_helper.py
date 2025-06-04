# core/execution/trade_execution_helper.py
from core.broker_api import submit_order_live
from bootstrap.simulate_trade_execution import simulate_trade_execution
from db.replay_buffer_sql import insert_replay_episode

class TradeExecutionHelper:
    def __init__(self, today, dry_run=False, prefix="[EXEC]"):
        self.today = today
        self.dry_run = dry_run
        self.prefix = prefix
        self.today_str = self.today.strftime("%Y-%m-%d")

    def execute(self, symbol, price, size, strategy_config, interval="day", mode="sim", trade_obj=None):
        """
        Execute a trade and compute result. Mode can be 'live' or 'sim'.
        """
        if mode == "live":
            return submit_order_live(trade_obj)  # assumes Trade class is passed
        else:
            result = simulate_trade_execution(trade_obj, self.today)
            return result

    def log_to_replay(self, result, strategy_config, interval):
        """
        Log trade to RL replay DB.
        """
        if not result:
            return
        episode = {
            "stock": result["symbol"],
            "date": self.today_str,
            "features": {},  # future: enrich and include features here
            "action": 1,
            "reward": result["reward"],
            "interval": interval,
            "strategy_config": strategy_config,
            "done": True
        }
        insert_replay_episode(episode)
