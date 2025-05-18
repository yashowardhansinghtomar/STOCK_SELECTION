# db/models.py

from datetime import datetime, date
from sqlalchemy import Column, String, Date, DateTime, Float, Integer, BigInteger, Boolean
from sqlalchemy.orm import declarative_base
from core.config import settings

Base = declarative_base()

class Instrument(Base):
    __tablename__ = 'instruments'
    instrument_token = Column(BigInteger, primary_key=True)
    exchange_token   = Column(BigInteger)
    tradingsymbol    = Column(String(50), unique=True, nullable=False)
    name             = Column(String)
    last_price       = Column(Float)
    expiry           = Column(Date)
    strike           = Column(Float)
    tick_size        = Column(Float)
    lot_size         = Column(Integer)
    instrument_type  = Column(String(20))
    segment          = Column(String(20))
    exchange         = Column(String(10))

class SkiplistStock(Base):
    __tablename__ = 'skiplist_stocks'
    stock       = Column(String(20), primary_key=True)
    date_added  = Column(DateTime, default=datetime.utcnow)
    reason      = Column(String)
    imported_at = Column(DateTime, default=datetime.utcnow)

class StockPriceHistory(Base):
    __tablename__ = 'stock_price_history'
    symbol      = Column(String(20), primary_key=True)
    date        = Column(Date, primary_key=True)
    open        = Column(Float, nullable=False)
    high        = Column(Float, nullable=False)
    low         = Column(Float, nullable=False)
    close       = Column(Float, nullable=False)
    adjclose    = Column(Float)
    volume      = Column(BigInteger, nullable=False)

class StockFeature(Base):
    __tablename__ = 'stock_features'
    stock             = Column(String(20), primary_key=True)
    date              = Column(Date, primary_key=True)
    sma_short         = Column(Float)
    sma_long          = Column(Float)
    rsi_thresh        = Column(Float)
    macd              = Column(Float)
    vwap              = Column(Float)
    atr_14            = Column(Float)
    bb_width          = Column(Float)
    macd_histogram    = Column(Float)
    price_compression = Column(Float)
    stock_encoded     = Column(Integer)
    volatility_10     = Column(Float)
    volume_spike      = Column(Boolean)
    vwap_dev          = Column(Float)

class StockFundamental(Base):
    __tablename__ = 'stock_fundamentals'
    stock           = Column(String(20), primary_key=True)
    name            = Column(String)
    sector          = Column(String)
    industry        = Column(String)
    pe_ratio        = Column(Float)
    pb_ratio        = Column(Float)
    debt_to_equity  = Column(Float)
    roe             = Column(Float)
    market_cap      = Column(Float)
    earnings_growth = Column(Float)
    imported_at     = Column(DateTime, default=datetime.utcnow)
    source          = Column(String, default="csv")
    date            = Column(Date)
    sma_short       = Column(Float)
    sma_long        = Column(Float)
    rsi_thresh      = Column(Float)
    stock_encoded   = Column(Integer)
    volume_proxy    = Column(BigInteger)

class StockEncoding(Base):
    __tablename__ = 'stock_encoding'
    stock           = Column(String(20), primary_key=True)
    encoded_value   = Column(Integer, unique=True)
    stock_encoded   = Column(Integer)

class Recommendation(Base):
    __tablename__ = 'recommendations'
    stock             = Column(String(20), primary_key=True)
    date              = Column(Date, primary_key=True)
    sma_short         = Column(Float)
    sma_long          = Column(Float)
    rsi_thresh        = Column(Float)
    total_return      = Column(Float)
    predicted_return  = Column(Float)
    confidence        = Column(Float)
    explanation       = Column(String)
    avg_trade_return  = Column(Float)
    imported_at       = Column(DateTime, default=datetime.utcnow)
    max_drawdown      = Column(Float)
    sharpe            = Column(Float)
    source            = Column(String)
    trade_count       = Column(Integer)
    trade_triggered   = Column(Integer)

class OpenPosition(Base):
    __tablename__ = 'open_positions'
    stock         = Column(String(20), primary_key=True)
    entry_date    = Column(DateTime)
    entry_price   = Column(Float)
    sma_short     = Column(Float)
    sma_long      = Column(Float)
    rsi_thresh    = Column(Float)

class PaperTrade(Base):
    __tablename__ = 'paper_trades'
    id              = Column(Integer, primary_key=True, autoincrement=True)
    stock           = Column(String(20), nullable=False)
    action          = Column(String(10))
    price           = Column(Float)
    quantity        = Column(Float)
    profit          = Column(Float)
    timestamp       = Column(DateTime, nullable=False, index=True)
    imported_at     = Column(DateTime, default=datetime.utcnow)
    signal_reason   = Column(String)
    source          = Column(String)
    strategy_config = Column(String)

class FilterModelPrediction(Base):
    __tablename__ = 'filter_model_predictions'
    date       = Column(Date, primary_key=True)
    stock      = Column(String(20), primary_key=True)
    score      = Column(Float)
    rank       = Column(Integer)
    confidence = Column(Float)
    decision   = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ParamModelPrediction(Base):
    __tablename__ = 'param_model_predictions'
    date            = Column(Date, primary_key=True)
    stock           = Column(String(20), primary_key=True)
    sma_short       = Column(Integer)
    sma_long        = Column(Integer)
    rsi_thresh      = Column(Float)
    confidence      = Column(Float)
    expected_sharpe = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow)
    sharpe          = Column(Float)  # Add this line

class PriceModelPrediction(Base):
    __tablename__ = 'price_model_predictions'
    date              = Column(Date, primary_key=True)
    stock             = Column(String(20), primary_key=True)
    predicted_price   = Column(Float)
    prediction_horizon = Column(Integer)
    model_version     = Column(String)
    confidence        = Column(Float)
    created_at        = Column(DateTime, default=datetime.utcnow)

class MLSelectedStock(Base):
    __tablename__ = settings.ml_selected_stocks_table

    id          = Column(Integer, primary_key=True, autoincrement=True)
    stock       = Column(String(20), nullable=False)
    source      = Column(String(50), nullable=True)
    imported_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f"<MLSelectedStock(stock={self.stock!r}, source={self.source!r}, imported_at={self.imported_at!r})>"
