from core.logger.logger import logger
# utils/file_io.py
import os
import pandas as pd

def load_dataframe(path, default_cols=None):
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
        if default_cols:
            return pd.DataFrame(columns=default_cols)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"❌ Failed to load {path}: {e}")
        return pd.DataFrame(columns=default_cols or [])

def save_dataframe(df, path, mode="w", header=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, mode=mode, header=header)
    logger.info(f"💾 Saved to {path}")
