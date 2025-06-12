# agents/portfolio_allocator.py

import pandas as pd
from core.logger.logger import logger
from core.config.config import settings
from core.data_provider.data_provider import load_data
from datetime import datetime

class PortfolioAllocatorAgent:
    def __init__(self, max_per_trade=10000, max_holdings=10, sector_limits=None):
        self.max_per_trade = max_per_trade
        self.max_holdings = max_holdings
        self.prefix = "🏦 [ALLOCATOR] "
        self.today = pd.to_datetime(datetime.now()).normalize()

        # Optional sector-level limits (e.g., {"IT": 0.3, "Finance": 0.2})
        self.sector_limits = sector_limits or {}

    def load_open_positions(self):
        df = load_data(settings.tables.open_positions)
        if df is None or df.empty:
            return pd.DataFrame(columns=["stock", "entry_price", "entry_date", "strategy_config"])
        return df

    def filter_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"{self.prefix}"+str(f"Evaluating {len(signals)} signals for allocation control..."))
        open_pos = self.load_open_positions()
        held = set(open_pos["stock"].tolist())

        remaining_budget = self.max_holdings - len(held)
        if remaining_budget <= 0:
            logger.warnings("Max holdings limit reached. No new positions allowed.", prefix=self.prefix)
            return pd.DataFrame()

        filtered = []
        for _, row in signals.iterrows():
            if row["stock"] in held:
                logger.info(f"{self.prefix}"+str(f"Already holding {row['stock']} — skipping"))
                continue

            if "confidence" in row and row["confidence"] < 0.1:
                logger.info(f"{self.prefix}"+str(f"Low confidence {row['stock']} — skipping"))
                continue

            # Sector-level check (optional, depends on enriched signals)
            if self.sector_limits:
                sector = row.get("sector")
                if sector and sector in self.sector_limits:
                    current_sector_count = (open_pos["sector"] == sector).sum()
                    max_sector = int(self.max_holdings * self.sector_limits[sector])
                    if current_sector_count >= max_sector:
                        logger.warnings(f"Sector cap reached for {sector}. Skipping {row['stock']}", prefix=self.prefix)
                        continue

            filtered.append(row)
            if len(filtered) >= remaining_budget:
                break

        logger.success(f"{len(filtered)} signals passed allocation rules.", prefix=self.prefix)
        return pd.DataFrame(filtered)
