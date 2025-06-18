# models/meta_strategy_selector.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools

from core.logger.logger import logger
from core.data_provider.data_provider import load_data, save_data
from core.config.config import settings

def load_combined_grid_data() -> pd.DataFrame:
    """
    Load and combine grid search results from all configured CSV paths.
    """
    dfs = []
    for path in settings.meta_grid_csv_paths:
        try:
            dfs.append(pd.read_csv(path))
        except Exception as e:
            logger.error(f"[ERROR] Failed to load {path}: {e}")
    if not dfs:
        raise RuntimeError("No grid-data CSVs loaded.")
    return pd.concat(dfs, ignore_index=True)

def train_meta_model():
    """
    Train a meta-model to predict strategy performance.
    Reads combined grid data, applies settings-driven train/test split,
    trains an RF regressor with settings-backed hyperparams,
    logs & saves the model and its metadata.
    """
    logger.start("🚀 Starting Meta-Strategy Model Training…")

    df = load_combined_grid_data()
    target = settings.meta_target_column
    df = df.dropna(subset=[target])
    df = df[df[target] > settings.meta_min_target_value]

    feature_cols = settings.meta_feature_columns  # e.g. ["sma_short","sma_long","rsi_thresh"]
    X = df[feature_cols]
    y = df[target]

    if len(X) < settings.meta_min_samples:
        logger.warning("[SKIP] Not enough samples to train meta model.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=settings.meta_test_size,
        random_state=settings.meta_random_state
    )

    # Instantiate with settings-backed hyperparams
    model = RandomForestRegressor(
        n_estimators=settings.meta_n_estimators,
        max_depth=settings.meta_max_depth,
        random_state=settings.meta_random_state,
        n_jobs=settings.meta_n_jobs
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    logger.success(f"✅ Meta-model trained. MSE: {mse:.4f}")

    # Save model + feature list
    save_data(
        pd.DataFrame([{
            "model_name": settings.meta_model_name,
            "features": feature_cols,
            "n_estimators": settings.meta_n_estimators,
            "max_depth": settings.meta_max_depth,
            "mse": mse,
            "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        }]),
        settings.meta_metadata_table
    )
    # Persist the model artifact
    from core.model_io import save_model
    save_model(settings.meta_model_name, (model, feature_cols))

def suggest_best_parameters(model) -> pd.DataFrame:
    """
    Given a trained meta-model, enumerate the cartesian product of
    settings-backed parameter ranges, predict their score, and return
    the top-N configs as a DataFrame.
    """
    sma_range = settings.meta_sma_short_range    # e.g. (5,50,5)
    long_range = settings.meta_sma_long_range    # e.g. (20,200,10)
    rsi_range = settings.meta_rsi_thresh_range   # e.g. (20,60,5)

    combos = itertools.product(
        range(*sma_range),
        range(*long_range),
        range(*rsi_range)
    )
    df = pd.DataFrame(combos, columns=feature_cols)
    df["predicted_score"] = model.predict(df[feature_cols])

    top_n = settings.meta_top_n
    return df.nlargest(top_n, "predicted_score")

if __name__ == "__main__":
    train_meta_model()
    # Optionally generate suggestions
    from core.model_io import load_model
    meta_model, feature_cols = load_model(settings.meta_model_name)
    best = suggest_best_parameters(meta_model)
    logger.info(f"Top {settings.meta_top_n} strategy configs:\n{best}")
