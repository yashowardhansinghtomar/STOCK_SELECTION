from prefect import flow
from core.logger.logger import logger
from db.postgres_manager import run_query
from core.time_context.time_context import get_simulation_date

@flow(name="Update RL Rewards", log_prints=True)
def update_rl_rewards_flow():
    today = get_simulation_date().date()
    logger.info(f"[RL REWARD FLOW] Running for {today}")

    # Ensure column exists
    run_query("""
        ALTER TABLE paper_trades
        ADD COLUMN IF NOT EXISTS rl_reward FLOAT;
    """, fetchall=False)

    # Update missed profit reward
    run_query("""
        UPDATE paper_trades SET rl_reward = 
            CASE 
                WHEN exit_price < max_price THEN GREATEST((max_price - exit_price) / exit_price, 0)
                ELSE 0
            END
        WHERE model_type = 'RL' AND date = CURRENT_DATE;
    """, fetchall=False)

    logger.info(f"[RL REWARD FLOW] Updated RL rewards for {today}")
