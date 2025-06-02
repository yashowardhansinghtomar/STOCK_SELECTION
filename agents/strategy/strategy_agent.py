from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from core.predict.predictor import predict_dual_model
from core.data_provider.data_provider import load_data
from core.config.config import settings
from core.logger.logger import logger
from db.models import StockFeatureDay as StockFeature
from core.time_context.time_context import get_simulation_date
from core.feature_engineering.feature_enricher_multi import enrich_multi_interval_features
from core.replay.replay_logger import log_replay_row
from db.db import SessionLocal

class StrategyAgent:
    def __init__(self, session: Session = None):
        self.session = session or SessionLocal()
        self.today = pd.to_datetime(get_simulation_date())
        self.today_str = self.today.strftime("%Y-%m-%d")
        self.today_date = self.today.date()

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

        enriched = enrich_multi_interval_features(stock, self.today, intervals=["day"])
        if enriched.empty:
            logger.warning(f"⚠️ No features available for {stock}")
            return {}

        preds = predict_dual_model(stock, enriched)
        if not preds:
            logger.warning(f"⚠️ No prediction result for {stock}")
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
            return {}

        top = preds[0]
        signal = {
            **top,
            "date": self.today_str,
            "stock": stock,
            "interval": "day",
            "strategy_config": {},  # JointPolicy handles config internally
            "exit_rule": {},
            "trade_triggered": int(top.get("trade_triggered", 1)),
            "source": "joint_policy",
            "explanation": "Signal from JointPolicy model"
        }

        logger.info(f"✅ JointPolicy signal for {stock}: {signal}")
        return signal

    def log_summary(self):
        logger.info("📊 Strategy summary: JointPolicy-only system — no legacy fallback used.")
