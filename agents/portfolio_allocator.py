from core.data_provider import load_data, save_data
from core.logger import logger
from core.config import settings
import pandas as pd
# agents/portfolio_allocator.py
class PortfolioAllocatorAgent:
    def __init__(self, total_capital=100000, max_per_trade_pct=0.1, confidence_weighted=True):
        self.total_capital = total_capital
        self.max_per_trade = total_capital * max_per_trade_pct
        self.confidence_weighted = confidence_weighted

    def allocate(self, recommendations: pd.DataFrame):
        recommendations["allocated_capital"] = (recommendations["confidence"] / recommendations["confidence"].sum()) * self.total_capital
        recommendations["allocated_capital"] = recommendations["allocated_capital"].clip(upper=settings.capital_per_trade)

        logger.info(f"🧮 Allocated capital: {recommendations[['stock','allocated_capital']]}")
        return recommendations


if __name__ == "__main__":
    agent = PortfolioAllocatorAgent(total_capital=50000)
    for conf in [0.5, 0.7, 0.9, 1.0]:
        amount = agent.allocate(confidence=conf)
        logger.info(f"Confidence: {conf:.2f} → Allocation: ₹{amount}")
