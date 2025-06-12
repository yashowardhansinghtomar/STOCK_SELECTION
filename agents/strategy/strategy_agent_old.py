# agents/strategy_agent.py
from datetime import datetime
import random
import pandas as pd
from sqlalchemy.orm import Session
from core.predict.predict_entry_exit_config import predict_entry_exit_config
from core.data_provider.data_provider import load_data
from core.config.config import settings
from core.logger.logger import logger
from core.predict.predictor import predict_dual_model
from core.skiplist.skiplist import add_to_skiplist
from db.models import StockFeatureDay as StockFeature, ParamModelPrediction
from core.model_io import load_model
from db.db import SessionLocal
from agents.time_series_agent import TimeSeriesAgent
from agents.strategy.rl_strategy_agent import RLStrategyAgent
from core.time_context.time_context import get_simulation_date
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.logger.logger import logger
from core.replay.replay_logger import log_replay_row

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
        self.today = pd.to_datetime(get_simulation_date())
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_date = self.today.date()

        self.model_dir = settings.model_dir
        self.ml_success = 0
        self.grid_fallback = 0
        self.ts_fallback = 0

        try:
            model_obj = load_model(f"{settings.filter_model_name}")
            self.filter_features = model_obj["features"]
            logger.info(f"✅ Loaded filter model features: {len(self.filter_features)} features")
        except Exception as e:
            logger.warnings(f"Could not load filter_model features: {e}")

            self.filter_features = []

    def fetch_features(self, stock: str) -> pd.DataFrame:
        recs = (
            self.session.query(StockFeature)
            .filter(StockFeature.stock == stock)
            .filter(StockFeature.date <= self.today_date)
            .all()
        )
        if not recs:
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in recs]).drop(columns="_sa_instance_state")
        return df

    def evaluate(self, stock: str) -> dict:
        logger.info(f"🔍 Evaluating {stock}")
        stock = stock.replace(".NS", "")

        # ✅ RL First
        try:
            rl_agent = RLStrategyAgent(model_name=f"ppo_{stock}", intervals=["day", "60minute", "15minute"])
            rl_sig = rl_agent.evaluate(stock)
            if rl_sig:
                logger.info(f"🧠 RL policy used successfully for {stock}")
                return rl_sig
        except Exception as e:
            logger.warnings(f"⚠️ RL failed for {stock}: {e}")

        # ✅ Then ML (Entry/Exit)
        signal = None
        for interval in ["day", "60minute", "15minute"]:
            enriched = enrich_multi_interval_features(stock, self.today, intervals=[interval])
            if enriched.empty or not is_valid_for_model(enriched, self.filter_features):
                continue

            config = predict_entry_exit_config(enriched)
            if not config or config.get("entry_signal") != 1:
                continue

            strategy_config = config.get("exit_rule", {})
            preds = predict_dual_model(stock, enriched)
            if not preds:
                continue

            top = preds[0]
            signal = {
                **top,
                "date": self.today_str,
                "stock": stock,
                "interval": interval,
                "strategy_config": strategy_config,
                "exit_rule": strategy_config,
                "trade_triggered": int(top.get("trade_triggered", 1)),
                "source": "entry_exit_model",
                "explanation": "ML signal via entry_exit_model + dual_model"
            }

            self.ml_success += 1
            self.session.merge(ParamModelPrediction(
                date=self.today_date,
                stock=stock,
                sma_short=strategy_config.get("sma_short"),
                sma_long=strategy_config.get("sma_long"),
                rsi_thresh=strategy_config.get("rsi_thresh"),
                confidence=top.get("confidence"),
                expected_sharpe=top.get("predicted_return"),
            ))
            break

        if signal:
            self.session.commit()
            logger.info(f"✅ ML signal for {stock}: {signal}")
            return signal

        # ✅ Then Time Series fallback
        if settings.ts_enabled:
            ts_agent = TimeSeriesAgent(stock, self.today)
            try:
                pred_price = ts_agent.predict()
            except Exception as e:
                import traceback
                logger.error(f"❌ TS model failed for {stock}: {e}\n{traceback.format_exc()}")
                pred_price = None

            if pred_price is not None:
                hist = load_data("stock_price_history")
                if hist is not None and "symbol" in hist.columns:
                    hist = hist[(hist["symbol"] == stock)]
                elif hist is not None and "stock" in hist.columns:
                    hist = hist[(hist["stock"] == stock)]
                hist["date"] = pd.to_datetime(hist["date"])
                hist["date"] = hist["date"].dt.tz_localize(None, nonexistent='shift_forward')

                cur_series = hist[hist["date"] <= self.today]["close"]
                if not cur_series.empty:
                    current_price = cur_series.iloc[-1]
                    gap = settings.ts_threshold
                    action = None
                    if pred_price > current_price * (1 + gap):
                        action = "buy"
                    elif pred_price < current_price * (1 - gap):
                        action = "sell"

                    if action:
                        self.ts_fallback += 1
                        logger.info(f"🔄 TS fallback {action.upper()} for {stock}: predicted {pred_price}")
                        return {
                            "date": self.today_str,
                            "stock": stock,
                            "signal": action,
                            "model": "ts_forecast",
                            "predicted_price": float(pred_price),
                            "confidence": None,
                            "strategy_config": {},
                            "exit_rule": {},
                            "trade_triggered": 1,
                            "source": "ts_fallback",
                            "explanation": "Time-series fallback signal"
                        }

        # Log skipped stock as "no_valid_signal"
        log_replay_row(
            stock=stock,
            action="none",
            reason="no_valid_signal",
            model="strategy_agent",
            prediction=None,
            confidence=None,
            signal=None,
            date=self.today_str
        )
        logger.info(f"🚫 No valid signal for {stock} — logged as skipped in replay.")

        return {}


    def _handle_grid_fallback(self, stock: str) -> dict:
        df = load_data(settings.tables.recommendations)
        if df is None or df.empty:
            return {}

        rec = df[(df.stock == stock) & (df.date == self.today_str)]
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
            "date": self.today_str,
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
            "trade_triggered": 1,
            "strategy_config": strategy_config
        }

    def log_summary(self):
        logger.info(
            f"📊 Strategy summary: {self.ml_success} ML, {self.grid_fallback} grid, {self.ts_fallback} TS fallbacks"
        )