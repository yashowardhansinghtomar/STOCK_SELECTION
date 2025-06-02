# training/train_sb3_ppo.py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.rl.sql_env import ODINSQLTradingEnv
from stable_baselines3.common.monitor import Monitor
import os

# ✅ 1. Wrap your env with Monitor + DummyVecEnv
def make_env():
    env = ODINSQLTradingEnv()
    env = Monitor(env)  # tracks rewards, length, etc.
    return env

vec_env = DummyVecEnv([make_env])  # for now single-process

# ✅ 2. Train
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs",
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
)

model.learn(total_timesteps=50000)

# ✅ 3. Save model
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/ppo_sb3_model")
