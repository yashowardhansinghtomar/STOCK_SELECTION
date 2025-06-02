# flows/train_rl_policy_flow.py

from prefect import flow
from core.logger.logger import logger
from agents.rl_ppo_trainer import retrain_ppo_if_ready

@flow(name="Train RL PPO Policy", log_prints=True)
def train_rl_policy_flow():
    logger.info("[RL TRAIN] Starting PPO policy retraining check...")
    retrain_ppo_if_ready()
    logger.success("[RL TRAIN] PPO policy retraining flow complete.")

if __name__ == "__main__":
    train_rl_policy_flow()
