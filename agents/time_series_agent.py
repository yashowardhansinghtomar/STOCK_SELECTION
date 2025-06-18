# agents/time_series_agent.py

from datetime import timedelta
import pandas as pd
import numpy as np
from pmdarima.arima import ARIMA
import warnings
import traceback

from core.config.config import settings
from core.data_provider.data_provider import load_data
from core.model_io import save_model, load_model
from core.logger.logger import logger

# 🔧 Always capture warnings as logs
warnings.simplefilter("always")

def warning_to_log(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"⚠️ [PYTHON WARNING] {category.__name__}: {message} @ {filename}:{lineno}")

warnings.showwarning = warning_to_log


class TimeSeriesAgent:
    BASE_MODEL_NAME = "ts_forecast"

    def __init__(self, symbol: str, ref_date: pd.Timestamp):
        self.symbol = symbol
        self.ref_date = pd.to_datetime(ref_date).date()
        self.order = settings.ts_order
        self.lookback = settings.ts_lookback_days

    def _get_hist(self) -> pd.Series:
        df = load_data(settings.price_history_table)
        if df.empty or "close" not in df.columns:
            return pd.Series(dtype=float)

        cols = [c for c in df.columns if c in {"symbol", "date", "close"}]
        df = df[cols].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

        df_sym = df[df["symbol"] == self.symbol].sort_values("date")
        if df_sym.empty:
            return pd.Series(dtype=float)

        cutoff = pd.to_datetime(self.ref_date) - timedelta(days=self.lookback)
        df_sym = df_sym[df_sym["date"] >= cutoff]

        series = df_sym.set_index("date")["close"].asfreq("D").ffill()
        series.index.name = None
        return pd.Series(series.values, index=series.index)

    def train_and_store(self):
        hist = self._get_hist()

        if isinstance(hist, pd.DataFrame):
            hist = hist.squeeze()

        hist = pd.to_numeric(hist, errors="coerce").dropna()
        if len(hist) < sum(self.order) + 1:
            logger.warning(f"⚠️ TS training skipped for {self.symbol}: insufficient data")
            return None

        logger.debug(f"[DEBUG] Type: {type(hist)}, Columns: {getattr(hist, 'columns', None)}")
        logger.debug(f"[DEBUG] Values Preview: {hist.head()}")
        logger.debug(f"[DEBUG] Values passed to ARIMA: {hist.values[:5]}")

        key = f"{self.BASE_MODEL_NAME}_{self.symbol}"
        try:
            model = ARIMA(order=self.order).fit(y=hist)
            save_model(key, model)
            logger.info(f"✅ Trained TS model: {key}")
            return model
        except Exception as e:
            logger.error(f"❌ TS training failed for {self.symbol}: {e}\n{traceback.format_exc()}")
            return None

    def predict(self) -> float | None:
        key = f"{self.BASE_MODEL_NAME}_{self.symbol}"
        try:
            model = load_model(key)
        except FileNotFoundError:
            model = self.train_and_store()

        if model is None:
            return None

        try:
            return float(model.predict(n_periods=1).iloc[0])
        except Exception as e:
            logger.error(f"❌ TS forecast failed for {self.symbol}: {e}\n{traceback.format_exc()}")
            return None
