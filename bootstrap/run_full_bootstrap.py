# bootstrap/run_full_bootstrap.py

from bootstrap.run_bootstrap import main as run_simulation
from bootstrap.bootstrap_trader import run_bootstrap_trader
from core.logger.logger import logger
import subprocess
import os

FILTER_MODEL_PATH = "models/filter_model.lgb"


def run_all():
    start_date = "2023-01-01"
    end_date = "2023-06-30"

    logger.info("🚀 Starting FULL BOOTSTRAP SYSTEM flow")

    # 0. Train filter model if missing
    if not os.path.exists(FILTER_MODEL_PATH):
        logger.warning("⚠️ Filter model not found. Training from scratch...")
        try:
            subprocess.run(["python", "-m", "models.train_stock_filter_model"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Filter model training failed: {e}")
            return
    else:
        logger.info("✅ Filter model already trained.")

    # 1. Run historical bootstrap over past N months
    logger.info("📜 Running historical simulation (Phase 0 → 2)...")
    try:
        subprocess.run(["python", "-m", "bootstrap.run_bootstrap", "--start", start_date, "--end", end_date], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Historical simulation failed: {e}")
        return

    # 2. Run today's live bootstrap (Phase 1+)
    logger.info("💹 Executing live bootstrap trading (ε-greedy)...")
    try:
        run_bootstrap_trader()
    except Exception as e:
        logger.error(f"❌ Live bootstrap trader failed: {e}")
        return

    # 3. (Optional) Trigger model training
    logger.info("🧠 Retraining joint + RL models from replay buffer...")
    try:
        subprocess.run(["python", "-m", "core.model_trainer.trainer"], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Model retraining skipped or failed: {e}")

    logger.info("✅ FULL BOOTSTRAP COMPLETE — Your system is ready to evolve!")


if __name__ == "__main__":
    run_all()
