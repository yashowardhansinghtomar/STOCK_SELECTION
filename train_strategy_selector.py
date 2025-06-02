from core.logger.logger import logger
# train_strategy_selector.py
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

INPUT_CSV = "training_data.csv"
MODEL_PATH = "model/best_strategy_model.pkl"

def train_strategy_selector():
    df = pd.read_csv(INPUT_CSV)

    feature_cols = [
        "pe_ratio", "de_ratio", "roe",
        "earnings_growth", "market_cap"
    ]
    label_cols = ["sma_short", "sma_long"]

    df = df.dropna(subset=feature_cols + label_cols)
    X = df[feature_cols]
    y = df[label_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((model, feature_cols), f)

    logger.success(f"✅ Strategy selector model trained and saved to {MODEL_PATH}")
    preds = model.predict(X_test)
    logger.info(f"🔍 Sample prediction:\n{pd.DataFrame(preds[:5], columns=label_cols)}")

if __name__ == "__main__":
    train_strategy_selector()
