from datetime import datetime
import random

import pandas as pd
from sqlalchemy.orm import Session
from core.data_provider import load_data
from core.config import settings
from core.logger import logger
from core.predictor import predict_dual_model
from core.skiplist import add_to_skiplist
from db.models import StockFeature, ParamModelPrediction
from core.grid_predictor import predict_grid_config
from core.model_io import load_model
from db.db import SessionLocal

from agents.time_series_agent import TimeSeriesAgent


def is_valid_for_model(df: pd.DataFrame, required: list, min_samples: int = 1) -> bool:
    if df is None or df.empty:
        return False
    missing = [c for c in required if c not in df.columns]
    if missing or df[required].dropna().shape[0] < min_samples:
        logger.debug(
            f"Invalid model input for features. Missing={missing}, "
            f"samples={df[required].dropna().shape[0] if not missing else 0}"
        )
        return False
    return True


class StrategyAgent:
    def __init__(self, session: Session = None):
        self.session = session or SessionLocal()
        self.today = datetime.now().date()

        self.model_dir = settings.model_dir
        self.ml_success = 0
        self.grid_fallback = 0
        self.ts_fallback = 0

        try:
            _, self.filter_features = load_model(f"{settings.filter_model_name}")
            logger.info(f"✅ Loaded filter model features: {len(self.filter_features)} features")
        except Exception as e:
            logger.warning(f"⚠️ Could not load filter_model features: {e}")
            self.filter_features = []

    def fetch_features(self, stock: str) -> pd.DataFrame:
        recs = (
            self.session.query(StockFeature)
            .filter(StockFeature.stock == stock)
            .filter(StockFeature.date <= self.today)
            .all()
        )
        if not recs:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in recs]).drop(columns="_sa_instance_state")
        return df

    def evaluate(self, stock: str) -> dict:
        logger.info(f"🔍 Evaluating {stock}")
        if stock.endswith(".NS"):
            stock = stock[:-3]

        df_feat = self.fetch_features(stock)
        if not df_feat.empty and is_valid_for_model(df_feat, self.filter_features):
            preds = predict_dual_model(stock, df_feat)
            if preds:
                top = preds[0]
                self.ml_success += 1
                logger.info(f"✅ ML signal for {stock}: {top}")

                self.session.merge(ParamModelPrediction(
                    date=self.today,
                    stock=stock,
                    sma_short=top['recommended_config'].get('sma_short'),
                    sma_long=top['recommended_config'].get('sma_long'),
                    rsi_thresh=top['recommended_config'].get('rsi_thresh'),
                    confidence=top.get('confidence'),
                    expected_sharpe=top.get('predicted_return'),
                ))
                self.session.commit()

                return {
                    **top,
                    "date": self.today.strftime("%Y-%m-%d"),
                    "stock": stock,
                    "trade_triggered": int(top.get("trade_triggered", 1)),
                    "strategy_config": top.get("recommended_config", {}),
                    "source": "ml"
                }

        result = self._handle_grid_fallback(stock)
        if result:
            result["source"] = "grid_fallback"
            return result

        if not settings.ts_enabled:
            return {}

        ts_agent = TimeSeriesAgent(stock, self.today)

        try:
            pred_price = ts_agent.predict()
        except Exception as e:
            import traceback
            logger.error(f"❌ TS model failed for {stock}: {e}\n{traceback.format_exc()}")
            return {}

        if pred_price is not None:
            hist = load_data("stock_price_history").copy()
            hist["date"] = pd.to_datetime(hist["date"])
            if hist["date"].dt.tz is not None:
                hist["date"] = hist["date"].dt.tz_convert(None)
                hist["date"] = hist["date"].dt.tz_localize(None)
            ts_today = pd.Timestamp(self.today)
            cur_series = hist[(hist.symbol == stock) & (hist.date <= ts_today)].close
            if not cur_series.empty:
                current_price = cur_series.iloc[-1]
                gap = settings.ts_threshold
                signal = None
                if pred_price > current_price * (1 + gap):
                    signal = "buy"
                elif pred_price < current_price * (1 - gap):
                    signal = "sell"

                if signal:
                    self.ts_fallback += 1
                    logger.info(f"🔄 TS fallback {signal.upper()} for {stock}: predicted {pred_price}")
                    return {
                        "date": self.today.strftime("%Y-%m-%d"),
                        "stock": stock,
                        "signal": signal,
                        "model": "ts_forecast",
                        "predicted_price": float(pred_price),
                        "predicted_return": None,
                        "total_return": None,
                        "avg_trade_return": None,
                        "sharpe": None,
                        "confidence": None,
                        "explanation": None,
                        "max_drawdown": None,
                        "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "trade_count": None,
                        "sma_short": None,
                        "sma_long": None,
                        "rsi_thresh": None,
                        "strategy_config": {},
                        "trade_triggered": int(1),
                        "source": "ts_fallback"
                    }

        return {}

    def _handle_grid_fallback(self, stock: str) -> dict:
        df = load_data(settings.recommendations_table)
        today = self.today.strftime("%Y-%m-%d")
        if df is None or df.empty:
            return {}

        rec = df[(df.stock == stock) & (df.date == today)]
        if rec.empty:
            return {}

        row = rec.iloc[0]
        strategy_config = {
            "sma_short": row.get("sma_short"),
            "sma_long": row.get("sma_long"),
            "rsi_thresh": row.get("rsi_thresh")
        }

        self.grid_fallback += 1
        logger.info(f"🔄 Grid fallback for {stock}: {strategy_config}")
        return {
            "date": today,
            "stock": stock,
            **strategy_config,
            "total_return": row.get("predicted_return"),
            "predicted_return": row.get("predicted_return"),
            "confidence": row.get("confidence"),
            "explanation": row.get("explanation"),
            "avg_trade_return": row.get("avg_trade_return"),
            "imported_at": row.get("imported_at"),
            "max_drawdown": row.get("max_drawdown"),
            "sharpe": row.get("sharpe"),
            "trade_count": row.get("trade_count"),
            "trade_triggered": int(1),
            "strategy_config": strategy_config
        }

    def log_summary(self):
        logger.info(
            f"📊 Strategy summary: {self.ml_success} ML, {self.grid_fallback} grid, {self.ts_fallback} TS fallbacks"
        )
