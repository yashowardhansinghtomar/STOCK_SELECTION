import pandas as pd
from core.logger import logger

class RiskManagementAgent:
    def __init__(self, max_drawdown=0.1, stop_loss=0.02, take_profit=0.05):
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def apply_risk_controls(self, open_positions: pd.DataFrame):
        updated_positions = []
        for idx, pos in open_positions.iterrows():
            entry_price = pos["entry_price"]
            current_price = pos["current_price"]

            pnl = (current_price - entry_price) / entry_price
            exit_reason = None

            if pnl <= -self.stop_loss:
                exit_reason = 'stop_loss'
            elif pnl >= self.take_profit:
                exit_reason = 'take_profit'

            if exit_reason:
                logger.info(f"🚨 Exiting {pos['stock']} due to {exit_reason}: PnL={pnl:.2%}")
            else:
                updated_positions.append(pos)

        return pd.DataFrame(updated_positions)
