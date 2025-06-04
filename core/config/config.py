# core/config.py

from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field, BaseModel
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from typing import ClassVar

def get_feature_columns(interval: str = "day") -> list:
    base = [
        "sma_short", "sma_long", "rsi_thresh",
        "macd", "vwap", "atr_14", "bb_width",
        "macd_histogram", "price_compression",
        "stock_encoded", "volatility_10",
        "volume_spike", "vwap_dev"
    ]
    if interval == "day":
        base += ["proxy_pe", "proxy_de_ratio", "proxy_roe", "proxy_growth", "proxy_market_cap"]
    return base

class RetrainConfig(BaseModel):
    paper_trades_threshold: int = 100
    training_data_threshold: int = 1000


class Settings(BaseSettings):
    # ─── Environment & DB ──────────────────────────────────────────────
    database_url: Optional[str] = None
    api_key: SecretStr = SecretStr("")
    api_secret: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    fetch_interval: int = Field(60, gt=0)
    REDIS_URL: str = Field("redis://redis:6379")

    # ─── Time-Series Fallback Settings ─────────────────────────────
    default_backfill_days: int     = 60
    ts_threshold: float        = 0.01       # 1% forecast move to trigger
    ts_order: Tuple[int,int,int] = (5,1,0)  # ARIMA(p,d,q)
    ts_lookback_days: int      = 252        # days of history for TS model
    ts_retrain_freq: str       = "weekly"   # how often to retrain TS models

    #─── Add new agent settings ───────────────────────────────────────────────────
    risk_management: dict = {
        "max_drawdown": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.05
    }
    # Add drift detection settings
    drift_detection_enabled: bool = True
    drift_features: list = ["sma_short", "sma_long", "rsi_thresh", "volatility_10", "macd_histogram"]

    # ─── RL Model Settings ─────────────────────────────────────────────
    policy_mode: str = "mix"       # or "rf", "rl"
    rl_allocation: int = 10        # percent

    # ─── ML Model Settings ─────────────────────────────────────────────    
    fallback_stocks: List[str] = Field(default_factory=lambda: [
        "RELIANCE", "ICICIBANK", "SBIN", "INFY", "LT"
    ])

    # ─── time series Agent switch ───────────────────────────────────────────────────
    ts_enabled: bool = True

    # ─── archiving switch ───────────────────────────────────────────────────
    enable_archiving: ClassVar[bool] = False

    # ─── fundamentals switch ───────────────────────────────────────────────────
    use_fundamentals: bool = False  # <- default True

    # ─── Table Names ───────────────────────────────────────────────────
    fundamentals_table: str        = "stock_fundamentals"
    feature_table_day: str         = "stock_features_day"
    feature_table_15m: str         = "stock_features_15m"
    feature_table_60m: str         = "stock_features_60m"
    feature_table_1m: str          = "stock_features_1m"
    recommendations_table: str     = "recommendations"
    open_positions_table: str      = "open_positions"
    trades_table: str              = "paper_trades"
    ml_selected_stocks_table: str  = "ml_selected_stocks"
    training_data_table: str       = "training_data"
    meta_training_table: str       = "meta_training"
    meta_metadata_table: str       = "meta_metadata"
    model_store_table: str         = "model_store"
    price_history_table: str       = "stock_price_history"
    instruments_table: str         = "instruments"
    skiplist_table: str            = "skiplist_stocks"
    encoding_table: str            = "stock_encoding"
    grid_params_table: str         = "grid_params"
    filter_model_predictions_table: str = "filter_model_predictions"
    param_model_predictions_table: str  = "param_model_predictions"
    price_model_predictions_table: str  = "price_model_predictions"
    selected_table: str            = "ml_selected_stocks"
    # ─── Agent & Model Names ──────────────────────────────────────────
    filter_model_name: str          = "filter_model"
    dual_classifier_model_name: str = "dual_model_classifier"
    dual_regressor_model_name: str  = "dual_model_regressor"
    meta_model_name: str            = "meta_model"
    strategy_type: str              = "dual"

    # ─── Strategy Defaults ───────────────────────────────────────────
    top_n: int = 5
    sma_short_window: int = 10
    sma_long_window: int = 30
    rsi_thresh: float = 30.0
    rsi_window: int = 14

    # ─── Directories & Files ─────────────────────────────────────────
    model_dir: Path = Path("models")
    log_dir: Path   = Path("logs")

    # ─── Logging ─────────────────────────────────────────────────────
    logger_name: str         = "app"
    log_level: str           = "INFO"
    console_log_level: str   = "INFO"
    file_log_level: str      = "DEBUG"
    log_format: str          = "[%(asctime)s] %(levelname)s %(message)s"
    json_logging: bool       = False
    json_logging_extra: bool = False

    # ─── Price‐Fetching & Caching ────────────────────────────────────
    price_fetch_interval: str = "day"
    price_fetch_days: int     = 2000
    price_cache_min_rows: int = 50


    # ─── Date Columns per Table ─────────────────────────────────────
    date_columns: Dict[str, List[str]] = {
        "stock_price_history": ["date"],
        "stock_features_day":  ["date"],
        "stock_features_15m":  ["date"],
        "stock_features_60m":  ["date"],
        "stock_features_1m":   ["date"],
        "stock_fundamentals":  ["imported_at"],
        "instruments":         ["expiry"],
        "skiplist_stocks":     ["date_added"],
        "recommendations":     ["date"],
        "paper_trades":        ["timestamp"],
        "meta_metadata":       ["trained_at"]
    }


    # ─── Backtest Defaults ──────────────────────────────────────────
    backtest_start: str = "2023-01-01"
    backtest_end: str   = "2024-01-01"

    # ─── Trade Execution ─────────────────────────────────────────────
    capital_per_trade: float = 10000.0

    # ─── Exit Logic Settings ────────────────────────────────────────
    exit_lookback_days: int = 200
    exit_ma_window:    int = 30

    # ─── Planner Agent Settings ─────────────────────────────────────
    max_eval: int                  = 50
    stock_whitelist: List[str]     = []
    indicator_columns: List[str] = [
        "sma_short", "sma_long", "rsi_thresh", "volume_spike",
        "volatility_10", "atr_14", "macd_histogram", "bb_width",
        "vwap_dev", "price_compression", "stock_encoded",
        "proxy_pe", "proxy_de_ratio", "proxy_roe",
        "proxy_growth", "proxy_market_cap"
    ]

    # ─── interval feature table map ─────────────────────────────────────
    interval_feature_table_map: Dict[str, str] = {
        "day": "stock_features_day",
        "15minute": "stock_features_15m",
        "60minute": "stock_features_60m",
        "minute": "stock_features_1m",  # if you're using "minute"
    }


    exit_feature_columns: List[str] = [
        "exit_kind", "stop_loss", "take_profit", "trail",
        "exit_sma_window", "max_holding_days"
    ]

    training_columns: List[str] = indicator_columns + exit_feature_columns + ["target"]

    # ─── ML Training & Split ────────────────────────────────────────
    test_size: float  = 0.2
    random_state: int = 42

    # ─── Meta‐Model Settings ───────────────────────────────────────
    meta_target_column: str         = "target"
    meta_min_target_value: float    = 0.0
    meta_feature_columns: List[str] = ["sma_short", "sma_long", "rsi_thresh"]
    meta_test_size: float           = 0.2
    meta_random_state: int          = 42
    meta_n_estimators: int          = 100
    meta_max_depth: Optional[int]   = None
    meta_n_jobs: int                = -1
    meta_grid_csv_paths: List[str]  = ["results/grid1.csv", "results/grid2.csv"]
    meta_sma_short_range: Tuple[int,int,int] = (5, 50, 5)
    meta_sma_long_range:  Tuple[int,int,int] = (20, 200, 10)
    meta_rsi_thresh_range:Tuple[int,int,int] = (20, 60, 5)
    meta_top_n: int                = 5
    meta_min_samples: int          = 10

    # ─── Retraining Thresholds & Model Names ───────────────────────
    retrain: RetrainConfig = RetrainConfig()

    model_names: Dict[str,str] = {
        "exit":   "exit_model",
        "filter": "filter_model",
        "dual":   "dual_model",
        "meta":   "meta_model"
    }

    # ─── Archiving Order ───────────────────────────────────────────
    archive_order: List[str] = [
        "stock_fundamentals",
        "stock_features_day",
        "stock_features_15m",
        "stock_features_60m",
        "stock_features_1m",
        "paper_trades",
        "training_data"
    ]


    # ─── Logical→Physical Map for load_data/save_data ──────────────
    table_map: Dict[str,str] = {
        "stock_fundamentals":         "stock_fundamentals",
        "stock_price_history":        "stock_price_history",
        "recommendations":            "recommendations",
        "open_positions":             "open_positions",
        "trades":                     "trades",
        "paper_trades":               "paper_trades",
        "ml_selected_stocks":         "ml_selected_stocks",
        "training_data":              "training_data",
        "meta_training":              "meta_training",
        "meta_metadata":              "meta_metadata",
        "model_store":                "model_store",
        "stock_encoding":             "stock_encoding",
        "instruments":                "instruments",
        "skiplist_stocks":            "skiplist_stocks",
        "grid_params":                "grid_params",
        "filter_model_predictions":   "filter_model_predictions",
        "param_model_predictions":    "param_model_predictions",
        "price_model_predictions":    "price_model_predictions",
        "stock_features_day":         "stock_features_day",
        "stock_features_15m":         "stock_features_15m",
        "stock_features_60m":         "stock_features_60m",
        "stock_features_1m":          "stock_features_1m"
    }


    # ─── Recommendation Columns ────────────────────────────────────────
    recommendation_columns: List[str] = [
        "stock",
        "date",
        "sma_short",
        "sma_long",
        "rsi_thresh",
        "total_return",
        "predicted_return",
        "confidence",
        "explanation",
        "avg_trade_return",
        "imported_at",
        "max_drawdown",
        "sharpe",
        "source",
        "trade_count",
        "trade_triggered",
    ]

    class Config:
        env_file    = ".env"
        extra       = "allow"

settings = Settings()

print("🧪 Loaded database_url =", settings.database_url)


if not settings.database_url:
    raise ValueError("❌ DATABASE_URL not set. Check your .env and docker-compose.yml")

if not settings.REDIS_URL:
    raise ValueError("❌ REDIS_URL not set. Check your .env and docker-compose.yml")