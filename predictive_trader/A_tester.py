import os
# 1. Train models
from predictive_trader.price_predictor_lstm import train_lstm_model
from predictive_trader.price_predictor_lgbm import train_lgbm_model

stock_list = ["RELIANCE", "TCS", "INFY", "ICICIBANK", "HDFCBANK"]

for stock in stock_list:
    train_lstm_model(stock)
    train_lgbm_model(stock)

# 2. Generate predictive signals
from predictive_trader.trade_signal_generator import generate_signals_for_list

generate_signals_for_list(stock_list)
