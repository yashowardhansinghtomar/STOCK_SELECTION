import pandas as pd
import vectorbt as vbt
from core.data_provider.data_provider import load_data
import optuna
from prefect import flow, task
from prefect.server.schemas.schedules import CronSchedule
from db.postgres_manager import run_query
from datetime import date

# ────────────────────────────────────────────────────────────────────────────────
# 1) Feature & Signal builders
# ────────────────────────────────────────────────────────────────────────────────
class FeatureBuilder:
    def __init__(self, price: pd.Series, fast_window: int, slow_window: int, rsi_window: int):
        self.price = price
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.rsi_window = rsi_window

    def build(self) -> pd.DataFrame:
        ma_fast = vbt.MA.run(self.price, window=self.fast_window).ma
        ma_slow = vbt.MA.run(self.price, window=self.slow_window).ma
        rsi     = vbt.RSI.run(self.price, window=self.rsi_window).rsi
        return pd.DataFrame({
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'rsi':     rsi
        })

class SignalGenerator:
    def __init__(self, feats: pd.DataFrame, rsi_threshold: float):
        self.feats = feats
        self.rsi_threshold = rsi_threshold

    def generate(self) -> pd.DataFrame:
        entries = self.feats['ma_fast'] > self.feats['ma_slow']
        exits   = self.feats['rsi'] > self.rsi_threshold
        return pd.DataFrame({'entries': entries, 'exits': exits})

# ────────────────────────────────────────────────────────────────────────────────
# 2) Tasks
# ────────────────────────────────────────────────────────────────────────────────
@task
def task_fetch_price(ticker: str, start: str, end: str) -> pd.Series:
    df = load_data(ticker, start=start, end=end)
    if 'close' not in df:
        raise ValueError(f"No 'close' column for {ticker}")
    return df['close']

task_build_features = task(lambda price, fw, sw, rw: FeatureBuilder(
    price, fast_window=fw, slow_window=sw, rsi_window=rw
).build())

task_generate_signals = task(lambda feats, rt: SignalGenerator(
    feats, rsi_threshold=rt
).generate())

task_backtest = task(lambda price, signals, init_cash: vbt.Portfolio.from_signals(
    price, signals['entries'], signals['exits'], init_cash=init_cash, freq='1D'
))

@task
def task_persist_signals(signals: pd.DataFrame, ticker: str):
    for ts, row in signals.iterrows():
        run_query(
            """
            INSERT INTO stock_recommendations (stock, date, entry_signal, exit_signal)
            VALUES (:stock, :date, :entry, :exit)
            ON CONFLICT (stock, date) DO UPDATE SET
              entry_signal = EXCLUDED.entry_signal,
              exit_signal  = EXCLUDED.exit_signal;
            """,
            params={
                'stock': ticker,
                'date':  ts.date(),
                'entry': bool(row['entries']),
                'exit':  bool(row['exits'])
            }
        )

@task
def task_execute(signals: pd.DataFrame):
    # TODO: replace with your broker/client calls
    for ts, row in signals.iterrows():
        if row['entries']:
            print(f"{ts.date()} → BUY signal")
        if row['exits']:
            print(f"{ts.date()} → SELL signal")
    return True

# ────────────────────────────────────────────────────────────────────────────────
# 3) Hyperparameter search (Optuna)
# ────────────────────────────────────────────────────────────────────────────────
def optimize_strategy(ticker: str, start: str, end: str, n_trials: int = 50) -> optuna.Study:
    df    = load_data(ticker, start=start, end=end)
    price = df['close']

    def objective(trial):
        fw  = trial.suggest_int('fast_window', 5, 20)
        sw  = trial.suggest_int('slow_window', 30, 100)
        rw  = trial.suggest_int('rsi_window', 10, 20)
        rt  = trial.suggest_int('rsi_threshold', 60, 90)

        feats = FeatureBuilder(price, fw, sw, rw).build()
        sigs  = SignalGenerator(feats, rt).generate()
        pf    = vbt.Portfolio.from_signals(price, sigs['entries'], sigs['exits'],
                                            init_cash=100_000, freq='1D')
        return pf.stats().get('SharpeRatio', 0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    return study

# ────────────────────────────────────────────────────────────────────────────────
# 4) Orchestration flow
# ────────────────────────────────────────────────────────────────────────────────
@flow(name="daily_trading_flow")

def daily_trading_flow(
    ticker: str = 'RELIANCE.NS',
    start: str  = '2024-01-01',
    end: str    = None,
    fast_window: int   = 10,
    slow_window: int   = 50,
    rsi_window: int    = 14,
    rsi_threshold: float = 70,
    init_cash: float   = 100_000
):
    if end is None:
        end = date.today().isoformat()

    price    = task_fetch_price(ticker, start, end)
    features = task_build_features(price, fast_window, slow_window, rsi_window)
    signals  = task_generate_signals(features, rsi_threshold)
    pf       = task_backtest(price, signals, init_cash)

    task_persist_signals(signals, ticker)
    task_execute(signals)

    return pf.stats()

if __name__ == "__main__":
    from prefect.deployments import Deployment

    Deployment.build_from_flow(
        flow=daily_trading_flow,
        name="daily-trading",
        work_pool_name="default",  # or your own work‐pool
        schedule=CronSchedule(cron="0 9 * * MON-FRI", timezone="Asia/Kolkata"),
    ).apply()
