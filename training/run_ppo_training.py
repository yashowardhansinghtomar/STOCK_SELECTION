# training/run_ppo_training.py

from core.rl.ppo_trainer import PPOTrainer
from core.rl.gym_env import ODINTradingEnv
from core.rl.sql_env import ODINSQLTradingEnv
import time

if __name__ == "__main__":
    try:
        env = ODINTradingEnv()
        print("✅ Using Redis-based ODINTradingEnv")
    except Exception as e:
        print(f"⚠️ Redis unavailable. Falling back to SQL: {e}")
        env = ODINSQLTradingEnv()

    trainer = PPOTrainer(env)

    print("🎯 Starting PPO training loop")
    for epoch in range(100):
        print(f"\nEpoch {epoch + 1}/100")
        trainer.collect_rollout(steps=500)
        trainer.train_step(batch_size=64)
        trainer.save_model()
        time.sleep(10)  # Optional sleep to simulate spaced training
