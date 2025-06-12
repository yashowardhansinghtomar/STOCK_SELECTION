# core/backtest_bt.py
import pandas as pd
from datetime import timedelta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from utils.time_utils import to_naive_utc, make_naive, make_naive_index
from core.config.config import settings
from core.data_provider.data_provider import fetch_stock_data
from core.config.strategy_config import StrategyConfig, ExitRule
import traceback

# ─────────────────────────────────────────────────────────
# Simple pandas implementations of SMA and RSI
# ─────────────────────────────────────────────────────────
def ta_sma(x, window):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()

def ta_rsi(x, window):
    series = pd.Series(x)
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

# ─────────────────────────────────────────────────────────
# Entry + Exit strategy based on injected config
# ─────────────────────────────────────────────────────────
class SMA_RSI_Exit(Strategy):
    cfg: StrategyConfig

    def init(self):
        c, d = self.cfg.sma_short, self.cfg.sma_long
        self.fast = self.I(ta_sma, self.data.Close, c)
        self.slow = self.I(ta_sma, self.data.Close, d)
        self.rsi = self.I(ta_rsi, self.data.Close, settings.rsi_window)
        self.entry_bar_index = None

    def next(self):
        price = self.data.Close[-1]
        bar_index = len(self.data.Close)

        fast_val = self.fast[-1]
        slow_val = self.slow[-1]
        rsi_val = self.rsi[-1]

        if not self.position:
            if crossover(self.fast, self.slow):
                print(f"🔍 {self.data.index[-1]} Crossover | Price={price:.2f} Fast={fast_val:.2f} Slow={slow_val:.2f} RSI={rsi_val:.2f}")
            if crossover(self.fast, self.slow) and rsi_val < self.cfg.rsi_entry:
                print(f"🚀 ENTRY: Buy at {price:.2f} | Bar {bar_index} | RSI={rsi_val:.2f}")
                self.buy()
                self.entry_bar_index = bar_index

        if not self.position:
            return

        rule = self.cfg.exit_rule
        pnl_pct = self.position.pl_pct

        if rule.kind == "fixed_pct":
            if (rule.stop_loss and pnl_pct <= -rule.stop_loss) or (rule.take_profit and pnl_pct >= rule.take_profit):
                print(f"💸 EXIT (Fixed): PnL={pnl_pct:.2%}")
                self.position.close()

        elif rule.kind == "trailing_pct":
            trail = price * (1 - rule.trail) if rule.trail else 0
            if pnl_pct <= -rule.stop_loss or price < trail:
                print(f"💸 EXIT (Trailing): PnL={pnl_pct:.2%}")
                self.position.close()

        elif rule.kind == "sma_cross" and rule.sma_window:
            sma_exit = self.I(ta_sma, self.data.Close, rule.sma_window)
            if crossover(sma_exit, self.fast):
                print(f"💸 EXIT (SMA Cross)")
                self.position.close()

        if rule.max_holding_days and self.entry_bar_index is not None:
            if bar_index - self.entry_bar_index >= rule.max_holding_days:
                print(f"⌛ EXIT (Max Hold Days): Held {bar_index - self.entry_bar_index} days")
                self.position.close()

# ─────────────────────────────────────────────────────────
# Main backtest wrapper
# ─────────────────────────────────────────────────────────
def run_backtest_config(stock: str, cfg: StrategyConfig, start=None, end=None, run_date=None) -> dict:
    if run_date:
        run_date = make_naive(pd.to_datetime(run_date))
        end = run_date
        start = run_date - timedelta(days=90)

    if start:
        start = make_naive(pd.to_datetime(start))
    if end:
        end = make_naive(pd.to_datetime(end))

    df = fetch_stock_data(stock, start=start or settings.backtest_start, end=end or settings.backtest_end)
    if df is None or df.empty:
        return {}

    df = df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.index = make_naive_index(pd.to_datetime(df.index))

    print(f"📌 DEBUG: df.index.tz = {df.index.tz}")
    print(f"📌 DEBUG: run_date tz = {run_date.tzinfo if run_date else 'None'}")

    SMA_RSI_Exit.cfg = cfg
    bt = Backtest(
        df,
        SMA_RSI_Exit,
        cash=settings.capital_per_trade,
        commission=getattr(settings, "backtest_commission", 0.001),
        trade_on_close=True,
        exclusive_orders=True
    )

    try:
        stats = bt.run()
    except Exception as e:
        print("❌ FULL BACKTEST ERROR:")
        traceback.print_exc()
        raise e

    return {
        "stock": stock,
        "total_return": stats.get("Return [%]", 0) / 100,
        "sharpe": stats.get("Sharpe Ratio", 0),
        "max_drawdown": stats.get("Max Drawdown [%]", 0) / 100,
        "avg_trade_return": stats.get("Avg. Trade %", 0) / 100,
        "trade_count": int(stats.get("# Trades", 0)),
    }

# ─────────────────────────────────────────────────────────
# Manual test
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    cfg = StrategyConfig(
        sma_short=10,
        sma_long=30,
        rsi_entry=55,
        exit_rule=ExitRule(
            kind="fixed_pct",
            stop_loss=0.03,
            take_profit=0.06,
            max_holding_days=10
        )
    )
    stats = run_backtest_config("RELIANCE.NS", cfg)
    print(stats)
