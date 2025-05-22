from pydantic import BaseModel
from typing import Literal, Optional

class ExitRule(BaseModel):
    kind: Literal["fixed_pct", "sma_cross", "time_stop", "trailing_pct"]
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trail: Optional[float] = None
    sma_window: Optional[int] = None
    max_holding_days: Optional[int] = None

class StrategyConfig(BaseModel):
    sma_short: int
    sma_long: int
    rsi_entry: float
    exit_rule: ExitRule
