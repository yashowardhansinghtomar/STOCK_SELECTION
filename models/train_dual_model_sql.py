# models/train_dual_model_sql.py
from core.model_io import save_model
from core.logger import logger
from core.data_provider import load_data
from core.model_io import save_model, load_model
from core.config import settings
import optuna
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

def _load_training(df_name: str):
    df = load_data(df_name)
    if df is None or df.empty:
        logger.error(f"❌ No training data found in '{df_name}'. Aborting.")
        return None, None, None
    X = df.drop(columns=["target", "date", "stock"])
    y = df["target"]
    return X, y, df


def train_dual_model(df_name: str = "training_data"):
    # Load data
    X, y, df = _load_training(df_name)
    if X is None:
        return
    # Split for classifier
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

    # Classifier HPO
    logger.info(f"🔍 Running classifier HPO for {settings.hpo_trials} trials...")
    study_clf = optuna.create_study(direction="maximize")
    study_clf.optimize(objective_class, n_trials=settings.hpo_trials)
    best_clf_params = study_clf.best_params
    logger.success(f"✅ Best classifier params: {best_clf_params}")
    clf_final = RandomForestClassifier(**best_clf_params, random_state=settings.random_state, n_jobs=-1)
    clf_final.fit(X_train_c, y_train_c)
    clf_preds = clf_final.predict(X_test_c)
    clf_acc = accuracy_score(y_test_c, clf_preds)
    logger.info(f"Classifier test accuracy: {clf_acc:.4f}")
    save_model("dual_classifier", {"model": clf_final, "features": list(X.columns), "params": best_clf_params})

    # Prepare for regressor using only positive cases
    df_reg = df.copy()
    df_reg["pred"] = clf_final.predict(X)
    pos_idx = df_reg[df_reg["pred"] == 1].index
    if pos_idx.empty:
        logger.warning("⚠️ No positive cases for regression. Skipping regressor training.")
        return
    X_reg = X.loc[pos_idx]
    y_reg = df_reg.loc[pos_idx, "return"] if "return" in df_reg.columns else df_reg.loc[pos_idx, "target"]

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

    # Regressor HPO
    logger.info(f"🔍 Running regressor HPO for {settings.hpo_trials} trials...")
    study_reg = optuna.create_study(direction="maximize")
    study_reg.optimize(objective_reg, n_trials=settings.hpo_trials)
    best_reg_params = study_reg.best_params
    logger.success(f"✅ Best regressor params: {best_reg_params}")
    reg_final = RandomForestRegressor(**best_reg_params, random_state=settings.random_state, n_jobs=-1)
    reg_final.fit(X_train_r, y_train_r)
    reg_preds = reg_final.predict(X_test_r)
    mse = mean_squared_error(y_test_r, reg_preds)
    logger.info(f"Regressor test MSE: {mse:.4f}")
    save_model("dual_regressor", {"model": reg_final, "features": list(X_reg.columns), "params": best_reg_params})

if __name__ == "__main__":
    train_dual_model()