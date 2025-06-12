# bootstrap/run_full_bootstrap.py

from bootstrap.run_bootstrap import main as run_simulation
from bootstrap.bootstrap_trader import run_bootstrap_trader
from core.logger.logger import logger
from core.market_calendar import get_trading_days
import subprocess
import os
from datetime import datetime
from core.model_trainer.trainer import train_models
from db.replay_buffer_sql import SQLReplayBuffer, policy_converged

FILTER_MODEL_PATH = "models/filter_model.lgb"

def run_all():
    phase0_start = "2023-01-01"
    phase0_end   = "2023-06-30"

    phase1_start = "2023-07-01"
    phase1_end   = "2023-09-30"

    logger.info("🚀 Starting FULL BOOTSTRAP SYSTEM flow")

    # 0. Skip recommendation generation if already present
    from db.db import SessionLocal
    from db.models import Recommendation

    def recommendations_exist():
        session = SessionLocal()
        try:
            return session.query(Recommendation).count() > 0
        finally:
            session.close()

    if recommendations_exist():
        logger.info("✅ Recommendations already exist. Skipping synthetic label generation.")
    else:
        logger.info("🧪 Generating backtest-based training data for filter model...")
        try:
            subprocess.run([os.sys.executable, "-m", "bootstrap.bootstrap_filter_training_data"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate synthetic recommendations: {e}")
            return

    # 1. Train filter model if missing
    if not os.path.exists(FILTER_MODEL_PATH):
        logger.warning("⚠️ Filter model not found. Training from scratch...")
        try:
            subprocess.run([os.sys.executable, "-m", "models.train_stock_filter_model"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Filter model training failed: {e}")
            return
    else:
        logger.info("✅ Filter model already trained.")

    # 2. Run historical bootstrap (Phase 0) — trading days only
    logger.info("📜 Running historical simulation (Phase 0)...")
    for sim_date in get_trading_days(datetime.strptime(phase0_start, "%Y-%m-%d"), datetime.strptime(phase0_end, "%Y-%m-%d")):
        try:
            logger.info(f"📅 Simulating Phase 0 day: {sim_date.date()}")
            subprocess.run([
                os.sys.executable, "-m", "bootstrap.run_bootstrap",
                "--start", sim_date.strftime("%Y-%m-%d"),
                "--end", sim_date.strftime("%Y-%m-%d")
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"❌ Phase 0 simulation failed on {sim_date.date()}: {e}")

    # 3. Run simulated live bootstrap (Phase 1+) on future dates
    logger.info("🕰️ Simulating live bootstrap Phase 1+ over future dates...")
    replay_buffer = SQLReplayBuffer()
    for idx, sim_date in enumerate(get_trading_days(datetime.strptime(phase1_start, "%Y-%m-%d"), datetime.strptime(phase1_end, "%Y-%m-%d"))):
        try:
            logger.info(f"🗓️ Simulating Phase 1+ live day: {sim_date.date()}")
            run_bootstrap_trader(sim_date)

            if policy_converged():
                logger.info(f"🤖 PPO policy converged by {sim_date.date()}")
            else:
                logger.info(f"🌀 PPO exploring on {sim_date.date()} (ε-greedy)")

            if sim_date.weekday() == 4:
                logger.info("🎯 Weekly training triggered...")
                train_models(replay_buffer, up_to_date=sim_date)

            if idx % 5 == 0:
                logger.info(f"💾 Saving replay buffer checkpoint for {sim_date.date()}")
                # replay_buffer.save_checkpoint(tag=f"wk_{sim_date.strftime('%Y%m%d')}")

        except Exception as e:
            logger.warning(f"⚠️ Live trader failed for {sim_date.date()}: {e}")

    # 4. Final retraining (optional)
    logger.info("🧠 Retraining joint + RL models from replay buffer...")
    try:
        subprocess.run([os.sys.executable, "-m", "core.model_trainer.trainer"], check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ Model retraining skipped or failed: {e}")

    logger.info("✅ FULL BOOTSTRAP COMPLETE — Your system is ready to evolve!")

if __name__ == "__main__":
    run_all()
