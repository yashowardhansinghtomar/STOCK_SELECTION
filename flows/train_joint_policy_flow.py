# flows/train_joint_policy_flow.py

from prefect import flow
from core.logger.logger import logger
from agents.joint_policy_trainer import train_joint_policy_if_ready

@flow(name="Train Joint Policy Model", log_prints=True)
def train_joint_policy_flow():
    logger.info("[JOINT TRAIN] Checking if joint model retraining is needed...")
    train_joint_policy_if_ready()
    logger.success("[JOINT TRAIN] Joint policy retraining flow complete.")

if __name__ == "__main__":
    train_joint_policy_flow()
