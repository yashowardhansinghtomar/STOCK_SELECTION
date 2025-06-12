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


class StrategyDefaults(BaseModel):
    top_n: int = 5
    sma_short_window: int = 10
    sma_long_window: int = 30
    rsi_thresh: float = 30.0
    rsi_window: int = 14


class LoggingConfig(BaseModel):
    logger_name: str = "app"
    log_level: str = "INFO"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    log_format: str = "[%(asctime)s] %(levelname)s %(message)s"
    json_logging: bool = False
    json_logging_extra: bool = False


class TableNames(BaseModel):
    fundamentals: str = "stock_fundamentals"
    features: Dict[str, str] = {
        "day": "stock_features_day",
        "15minute": "stock_features_15m",
        "60minute": "stock_features_60m",
        "minute": "stock_features_1m"
    }
    predictions: Dict[str, str] = {
        "filter": "filter_model_predictions",
        "param": "param_model_predictions",
        "price": "price_model_predictions"
    }
    recommendations: str = "recommendations"
    open_positions: str = "open_positions"
    trades: str = "paper_trades"
    ml_selected: str = "ml_selected_stocks"
    training_data: str = "training_data"
    meta_training: str = "meta_training"
    meta_metadata: str = "meta_metadata"
    model_store: str = "model_store"
    price_history: str = "stock_price_history"
    instruments: str = "instruments"
    skiplist: str = "skiplist_stocks"
    encoding: str = "stock_encoding"
    grid_params: str = "grid_params"


class FeatureGroupConfig(BaseModel):
    indicator_columns: List[str] = [
        "sma_short", "sma_long", "rsi_thresh", "volume_spike",
        "volatility_10", "atr_14", "macd_histogram", "bb_width",
        "vwap_dev", "price_compression", "stock_encoded",
        "proxy_pe", "proxy_de_ratio", "proxy_roe",
        "proxy_growth", "proxy_market_cap"
    ]
    exit_feature_columns: List[str] = [
        "exit_kind", "stop_loss", "take_profit", "trail",
        "exit_sma_window", "max_holding_days"
    ]
    training_columns: List[str] = Field(default_factory=lambda: [
        "sma_short", "sma_long", "rsi_thresh", "volume_spike",
        "volatility_10", "atr_14", "macd_histogram", "bb_width",
        "vwap_dev", "price_compression", "stock_encoded",
        "proxy_pe", "proxy_de_ratio", "proxy_roe",
        "proxy_growth", "proxy_market_cap",
        "exit_kind", "stop_loss", "take_profit", "trail",
        "exit_sma_window", "max_holding_days", "target"
    ])
    recommendation_columns: List[str] = [
        "stock", "date", "sma_short", "sma_long", "rsi_thresh",
        "total_return", "predicted_return", "confidence",
        "explanation", "avg_trade_return", "imported_at",
        "max_drawdown", "sharpe", "source", "trade_count",
        "trade_triggered"
    ]


class Settings(BaseSettings):
    database_url: Optional[str] = None
    api_key: SecretStr = SecretStr("")
    api_secret: Optional[SecretStr] = None
    access_token: Optional[SecretStr] = None
    REDIS_URL: str = Field("redis://redis:6379")
    fetch_interval: int = Field(60, gt=0)

    # Modules
    strategy_defaults: StrategyDefaults = StrategyDefaults()
    logging: LoggingConfig = LoggingConfig()
    tables: TableNames = TableNames()
    features: FeatureGroupConfig = FeatureGroupConfig()
    retrain: RetrainConfig = RetrainConfig()

    # Misc
    policy_mode: str = "mix"
    rl_allocation: int = 10
    fallback_stocks: List[str] = Field(default_factory=lambda: ["RELIANCE", "ICICIBANK", "SBIN", "INFY", "LT"])
    drift_detection_enabled: bool = True
    drift_features: list = ["sma_short", "sma_long", "rsi_thresh", "volatility_10", "macd_histogram"]

    interval_feature_table_map: Dict[str, str] = TableNames().features
    date_columns: Dict[str, List[str]] = {
        "stock_price_history": ["date"],
        "stock_features_day": ["date"],
        "stock_features_15m": ["date"],
        "stock_features_60m": ["date"],
        "stock_features_1m": ["date"],
        "stock_fundamentals": ["imported_at"],
        "instruments": ["expiry"],
        "skiplist_stocks": ["date_added"],
        "recommendations": ["date"],
        "paper_trades": ["timestamp"],
        "meta_metadata": ["trained_at"]
    }

    enable_archiving: ClassVar[bool] = False
    use_fundamentals: bool = False
    ts_enabled: bool = True

    backtest_commission: float = 0.001
    backtest_start: str = "2023-01-01"
    backtest_end: str = "2024-01-01"
    rsi_window: int = 14

    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")

    price_fetch_interval: str = "day"
    price_fetch_days: int = 2000
    price_cache_min_rows: int = 50
    capital_per_trade: float = 10000.0

    test_size: float = 0.2
    random_state: int = 42

    meta_target_column: str = "target"
    meta_min_target_value: float = 0.0
    meta_feature_columns: List[str] = ["sma_short", "sma_long", "rsi_thresh"]
    meta_test_size: float = 0.2
    meta_random_state: int = 42
    meta_n_estimators: int = 100
    meta_max_depth: Optional[int] = None
    meta_n_jobs: int = -1
    meta_grid_csv_paths: List[str] = ["results/grid1.csv", "results/grid2.csv"]
    meta_sma_short_range: Tuple[int, int, int] = (5, 50, 5)
    meta_sma_long_range: Tuple[int, int, int] = (20, 200, 10)
    meta_rsi_thresh_range: Tuple[int, int, int] = (20, 60, 5)
    meta_top_n: int = 5
    meta_min_samples: int = 10

    model_names: Dict[str, str] = {
        "exit": "exit_model",
        "filter": "filter_model",
        "dual": "dual_model",
        "meta": "meta_model"
    }

    archive_order: List[str] = [
        "stock_fundamentals", "stock_features_day", "stock_features_15m",
        "stock_features_60m", "stock_features_1m", "paper_trades",
        "training_data"
    ]

    table_map: Dict[str, str] = {**TableNames().features, **TableNames().predictions, **{
        "stock_fundamentals": "stock_fundamentals",
        "stock_price_history": "stock_price_history",
        "recommendations": "recommendations",
        "open_positions": "open_positions",
        "trades": "trades",
        "paper_trades": "paper_trades",
        "ml_selected_stocks": "ml_selected_stocks",
        "training_data": "training_data",
        "meta_training": "meta_training",
        "meta_metadata": "meta_metadata",
        "model_store": "model_store",
        "stock_encoding": "stock_encoding",
        "instruments": "instruments",
        "skiplist_stocks": "skiplist_stocks",
        "grid_params": "grid_params"
    }}

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()


def get_feature_table(interval: str) -> str:
    table_map = settings.interval_feature_table_map
    if interval not in table_map:
        raise ValueError(f"Unknown interval '{interval}'. Available: {list(table_map.keys())}")
    return table_map[interval]


print("\U0001F9EA Loaded database_url =", settings.database_url)

if not settings.database_url:
    raise ValueError("\u274C DATABASE_URL not set. Check your .env and docker-compose.yml")

if not settings.REDIS_URL:
    raise ValueError("\u274C REDIS_URL not set. Check your .env and docker-compose.yml")
