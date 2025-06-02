import pandas as pd
from db.postgres_manager import run_query, SessionLocal
from core.logger.logger import logger
from core.system_state import get_system_config, update_system_config
from prefect import flow
from core.time_context.time_context import get_simulation_date
from core.policy.policy_manager import choose_best_policy
from sqlalchemy import Table, Column, Integer, String, Date, Float, JSON, MetaData, TIMESTAMP, text, insert
from agents.allocator_agent import AllocatorAgent

# Define table metadata
metadata = MetaData()
daily_policy_choices_table = Table(
    "daily_policy_choices",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("date", Date, nullable=False),
    Column("stock", String, nullable=False),
    Column("interval", String, nullable=False, default='day'),
    Column("chosen_policy_type", String, nullable=False),
    Column("parameters", JSON, nullable=False),
    Column("source", String, nullable=False),
    Column("confidence_score", Float),
    Column("notes", String),
    Column("created_at", TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")),
)

def save_policy_choice(
    date: str,
    stock: str,
    interval: str,
    policy_type: str,
    parameters: dict,
    source: str,
    confidence: float = None,
    notes: str = None
):
    with SessionLocal() as session:
        stmt = insert(daily_policy_choices_table).values(
            date=date,
            stock=stock,
            interval=interval,
            chosen_policy_type=policy_type,
            parameters=parameters,
            source=source,
            confidence_score=confidence,
            notes=notes,
        ).on_conflict_do_nothing(index_elements=["date", "stock", "interval"])
        session.execute(stmt)
        session.commit()


@flow(name="Daily Policy Chooser", log_prints=True)
def policy_chooser_flow():
    today = get_simulation_date().date()
    logger.info(f"[PolicyChooser] Running for {today}")

    choose_best_policy(today)
    AllocatorAgent().run()

    logger.success("[PolicyChooser] Completed.")


# === Sharpe-based Allocator Logic ===

def get_sharpe(model_type: str, days: int = 7) -> float:
    query = f"""
    SELECT date, SUM(pnl) AS daily_pnl
    FROM paper_trades
    WHERE model_type = '{model_type}' AND date >= CURRENT_DATE - INTERVAL '{days} days'
    GROUP BY date
    ORDER BY date;
    """
    df = run_query(query)
    if df.empty or df['daily_pnl'].nunique() <= 1:
        return 0.0
    returns = df['daily_pnl'].pct_change().dropna()
    return returns.mean() / returns.std()

def policy_chooser():
    sharpe_rl = get_sharpe('RL')
    sharpe_rf = get_sharpe('RF')
    logger.info(f"[POLICY CHOOSER] RL Sharpe: {sharpe_rl:.3f} vs RF: {sharpe_rf:.3f}")

    delta = sharpe_rl - sharpe_rf
    current = get_current_allocation()
    new_alloc = current

    if delta > -0.05:
        new_alloc = min(100, current + 10)
        reason = "RL performing better or equal"
    elif delta < -0.15:
        new_alloc = max(0, current - 10)
        reason = "RL significantly underperforming"
    else:
        reason = "RL slightly underperforming — holding allocation"

    if new_alloc != current:
        set_current_allocation(new_alloc)
        logger.info(f"[POLICY CHOOSER] RL allocation changed from {current}% to {new_alloc}% ({reason})")
    else:
        logger.info(f"[POLICY CHOOSER] RL allocation unchanged at {current}% ({reason})")

def get_current_allocation() -> int:
    config = get_system_config()
    return int(config.get("rl_allocation", 10))

def set_current_allocation(percent: int):
    update_system_config({"rl_allocation": percent})
    logger.info(f"[POLICY CHOOSER] RL allocation set to {percent}%")

if __name__ == "__main__":
    policy_chooser()
