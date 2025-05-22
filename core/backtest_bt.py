import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from core.config import settings
from core.data_provider import fetch_stock_data
from core.strategy_config import StrategyConfig, ExitRule

# ─────────────────────────────────────────────────────────
# Simple pandas implementations of SMA and RSI
# ─────────────────────────────────────────────────────────
def ta_sma(x, window):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()

def ta_rsi(x, window):
    series = pd.Series(x)
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

# ─────────────────────────────────────────────────────────
# Entry + Exit strategy based on injected config
# ─────────────────────────────────────────────────────────
class SMA_RSI_Exit(Strategy):
    cfg: StrategyConfig  # injected at runtime

    def init(self):
        c, d = self.cfg.sma_short, self.cfg.sma_long
        self.fast = self.I(ta_sma, self.data.Close, c)
        self.slow = self.I(ta_sma, self.data.Close, d)
        self.rsi  = self.I(ta_rsi, self.data.Close, settings.rsi_window)

    def next(self):
        price = self.data.Close[-1]

        # Entry rule
        if crossover(self.fast, self.slow) and self.rsi[-1] < self.cfg.rsi_entry:
            self.buy()

        # Exit rule
        if not self.position:
            return

        rule = self.cfg.exit_rule

        if rule.kind == "fixed_pct":
            pnl = (price - self.position.entry_price) / self.position.entry_price
            if (rule.stop_loss and pnl <= -rule.stop_loss) or (rule.take_profit and pnl >= rule.take_profit):
                self.position.close()

        elif rule.kind == "trailing_pct":
            trail = price * (1 - rule.trail) if rule.trail else 0
            if self.position.pl_pct <= -rule.stop_loss or price < trail:
                self.position.close()

        elif rule.kind == "sma_cross":
            if rule.sma_window:
                sma_exit = self.I(ta_sma, self.data.Close, rule.sma_window)
                if crossover(sma_exit, self.fast):  # fast falls below exit SMA
                    self.position.close()

        if rule.max_holding_days and self.bar_index - self.position.open_bar >= rule.max_holding_days:
            self.position.close()

# ─────────────────────────────────────────────────────────
# Main backtest wrapper using StrategyConfig
# ─────────────────────────────────────────────────────────
def run_backtest_config(stock: str, cfg: StrategyConfig, start=None, end=None) -> dict:
    df = fetch_stock_data(stock, start=start or settings.backtest_start, end=end or settings.backtest_end)
    if df is None or df.empty:
        return {}

    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })[['Open', 'High', 'Low', 'Close', 'Volume']]

    SMA_RSI_Exit.cfg = cfg
    bt = Backtest(
        df,
        SMA_RSI_Exit,
        cash=settings.capital_per_trade,
        commission=settings.backtest_commission,
        trade_on_close=True,
        exclusive_orders=True
    )

    stats = bt.run()
    return {
        "stock": stock,
        "total_return": stats.get("Return [%]", 0) / 100,
        "sharpe": stats.get("Sharpe Ratio", 0),
        "max_drawdown": stats.get("Max Drawdown [%]", 0) / 100,
        "avg_trade_return": stats.get("Avg. Trade %", 0) / 100,
        "trade_count": int(stats.get("# Trades", 0)),
    }

# ─────────────────────────────────────────────────────────
# Example run for manual testing
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    example_cfg = StrategyConfig(
        sma_short=10,
        sma_long=30,
        rsi_entry=30.0,
        exit_rule=ExitRule(
            kind="fixed_pct",
            stop_loss=0.03,
            take_profit=0.06,
            max_holding_days=10
        )
    )
    stats = run_backtest_config("RELIANCE.NS", example_cfg)
    print(stats)
