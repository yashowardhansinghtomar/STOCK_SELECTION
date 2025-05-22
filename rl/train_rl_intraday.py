# rl/train_rl_intraday.py

from rl.train_rl_agent import main as train_main

if __name__ == "__main__":
    import sys
    sys.argv += ["--interval", "15minute", "--name", "ppo_intraday"]
    train_main()
