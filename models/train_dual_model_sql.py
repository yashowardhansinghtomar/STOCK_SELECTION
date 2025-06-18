# models/train_dual_model_sql.py

import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

from core.logger.logger import logger
from core.config.config import settings
from core.data_provider.data_provider import load_data
from core.model_io import save_model


def _load_training(df_name: str):
    df = load_data(df_name)
    if df is None or df.empty:
        logger.error(f"❌ No training data found in '{df_name}'. Aborting.")
        return None, None, None

    missing = [col for col in settings.training_columns if col not in df.columns]
    if missing:
        logger.error(f"❌ Missing required columns in training data: {missing}")
        return None, None, None

    X = df[settings.training_columns].drop(columns=["target"])
    y = df["target"]
    return X, y, df


def train_dual_model(df_name: str = "training_data"):
    logger.start("🧠 Training dual-model (entry + exit config aware)…")

    X, y, df = _load_training(df_name)
    if X is None:
        return

    # ─── Classifier ────────────────────────────────────────
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state
    )

    def objective_class(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": settings.random_state,
            "n_jobs": -1
        }
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_c, y_train_c)
        preds = clf.predict(X_test_c)
        return accuracy_score(y_test_c, preds)

    logger.info(f"🔍 Running classifier HPO for {settings.meta_top_n} trials...")
    study_clf = optuna.create_study(direction="maximize")
    study_clf.optimize(objective_class, n_trials=settings.meta_top_n)

    best_clf_params = study_clf.best_params
    logger.success(f"✅ Best classifier params: {best_clf_params}")

    clf_final = RandomForestClassifier(**best_clf_params, random_state=settings.random_state, n_jobs=-1)
    clf_final.fit(X_train_c, y_train_c)
    acc = accuracy_score(y_test_c, clf_final.predict(X_test_c))
    logger.info(f"📊 Classifier accuracy: {acc:.4f}")

    save_model(settings.model_names["dual"] + "_classifier", {
        "model": clf_final,
        "features": list(X.columns),
        "params": best_clf_params
    }, meta={
        "type": "classifier",
        "algo": "RandomForest",
        "accuracy": round(acc, 4),
        "trained_at": str(pd.Timestamp.now()),
        "features": list(X.columns),
        "params": best_clf_params
    })

    # ─── Regressor ──────────────────────────────────────────
    df["pred"] = clf_final.predict(X)
    pos_idx = df[df["pred"] == 1].index

    if pos_idx.empty:
        logger.warning("⚠️ No positive predictions for regressor training. Skipping.")
        return

    X_reg = X.loc[pos_idx]
    if "return" in df.columns:
        y_reg = df.loc[pos_idx, "return"]
    elif "profit" in df.columns:
        y_reg = df.loc[pos_idx, "profit"]
    else:
        logger.error("❌ No return or profit column found for regressor.")
        return

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=settings.test_size, random_state=settings.random_state
    )

    def objective_reg(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": settings.random_state,
            "n_jobs": -1
        }
        reg = RandomForestRegressor(**params)
        reg.fit(X_train_r, y_train_r)
        preds = reg.predict(X_test_r)
        return -mean_squared_error(y_test_r, preds)

    logger.info(f"🔍 Running regressor HPO for {settings.meta_top_n} trials...")
    study_reg = optuna.create_study(direction="maximize")
    study_reg.optimize(objective_reg, n_trials=settings.meta_top_n)

    best_reg_params = study_reg.best_params
    logger.success(f"✅ Best regressor params: {best_reg_params}")

    reg_final = RandomForestRegressor(**best_reg_params, random_state=settings.random_state, n_jobs=-1)
    reg_final.fit(X_train_r, y_train_r)
    mse = mean_squared_error(y_test_r, reg_final.predict(X_test_r))
    logger.info(f"📉 Regressor MSE: {mse:.4f}")

    save_model(settings.model_names["dual"] + "_regressor", {
        "model": reg_final,
        "features": list(X_reg.columns),
        "params": best_reg_params
    }, meta={
        "type": "regressor",
        "algo": "RandomForest",
        "mse": round(mse, 6),
        "trained_at": str(pd.Timestamp.now()),
        "features": list(X_reg.columns),
        "params": best_reg_params
    })

    logger.success("✅ Dual model training complete.")


if __name__ == "__main__":
    train_dual_model()
