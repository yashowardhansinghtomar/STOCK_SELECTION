# core/backtest_bt.py

import pandas as pd
from backtesting import Strategy, Backtest
from backtesting.lib import crossover
from core.config import settings        # ← ADD THIS
from core.data_provider import fetch_stock_data

# Simple pandas implementations of SMA and RSI
def ta_sma(x, window):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()

def ta_rsi(x, window):
    # Wilder’s RSI
    series = pd.Series(x)
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

class SMA_RSI(Strategy):
    """
    SMA crossover entry with RSI‐based exit.
    """
    sma_short  = settings.sma_short_window
    sma_long   = settings.sma_long_window
    rsi_window = settings.rsi_window
    rsi_thresh = settings.rsi_thresh

    def init(self):
        close = self.data.Close
        self.fast = self.I(ta_sma, close, self.sma_short)
        self.slow = self.I(ta_sma, close, self.sma_long)
        self.rsi  = self.I(ta_rsi, close, self.rsi_window)

    def next(self):
        if crossover(self.fast, self.slow):
            self.buy()
        elif self.rsi[-1] > self.rsi_thresh:
            self.position.close()

def run_backtest(ticker: str,
                 start: str = settings.backtest_start,
                 end:   str = settings.backtest_end):
    """
    Fetch OHLCV data, run the SMA_RSI strategy, and return performance stats.
    """
    df = fetch_stock_data(ticker, start=start, end=end)
    if df is None or df.empty:
        return None

    # backtesting.py expects these column names
    df = df.rename(columns={
        'open':   'Open',
        'high':   'High',
        'low':    'Low',
        'close':  'Close',
        'volume': 'Volume'
    })[['Open','High','Low','Close','Volume']]

    bt = Backtest(
        df,
        SMA_RSI,
        cash=settings.backtest_cash,
        commission=settings.backtest_commission,
        trade_on_close=True,
        exclusive_orders=True
    )
    return bt.run()

if __name__ == '__main__':
    # Windows‐safe entry point for multiprocessing
    stats = run_backtest('RELIANCE.NS')
    print(stats)
