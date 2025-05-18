# hpo.py
from models.train_stock_filter_model import train_stock_filter_model
from models.train_dual_model_sql import train_dual_model
from models.meta_strategy_selector import train_meta_model

if __name__ == "__main__":
    train_stock_filter_model()
    train_dual_model()
    train_meta_model()
