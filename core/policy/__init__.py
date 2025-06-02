# core/policy/__init__.py

from core.policy.rl_policy import RLPolicyModel
from models.joint_policy import JointPolicyModel
from core.system_state import get_system_config
from core.logger.logger import logger
import pandas as pd

# Load models at module level
_rl = RLPolicyModel()
_joint = JointPolicyModel()


try:
    _rl.load()
except Exception as e:
    logger.warning(f"[POLICY INIT] RL model not loaded: {e}")

try:
    _joint.load()
except Exception as e:
    logger.warning(f"[POLICY INIT] Joint model not loaded: {e}")

def choose_policy_model():
    config = get_system_config()
    mode = config.get("policy_mode", "mix").lower()

    if mode == "rl":
        return _rl
    elif mode == "joint":
        return _joint
    else:
        logger.warning(f"[POLICY INIT] Unknown policy_mode: {mode}, falling back to RL.")
        return _rl

