# flows/auto_pipeline.py
from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from prefect.server.schemas.schedules import IntervalSchedule

from core.config.config import settings
from db.db import get_session  # SQLAlchemy session factory
from sqlalchemy import text

PIPELINE_NAME = "self_learning_pipeline"

def get_last_date(session):
    row = session.execute(
        text("SELECT last_date_processed FROM pipeline_metadata WHERE pipeline_name=:name"),
        {"name": PIPELINE_NAME}
    ).fetchone()
    return row[0] if row else None

def update_last_date(session, new_date):
    session.execute(text("""
      INSERT INTO pipeline_metadata(pipeline_name, last_date_processed)
      VALUES (:name, :date)
      ON CONFLICT (pipeline_name) DO UPDATE
        SET last_date_processed = EXCLUDED.last_date_processed
    """), {"name": PIPELINE_NAME, "date": new_date})
    session.commit()

@task(retries=3, retry_delay_seconds=60)
def ingest_data(run_date: date):
    from core.data_provider.data_provider import fetch_stock_data, save_data
    logger = get_run_logger()
    # assume fundamentals pre‐loaded
    from core.data_provider.data_provider import load_data
    fundamentals = load_data(settings.fundamentals_table)
    for stock in fundamentals["stock"].unique():
        # fetch exactly one day of data
        df = fetch_stock_data(stock, end=run_date, days=1)
        if not df.empty:
            save_data(df.reset_index(), settings.price_history_table)
            logger.info(f"[{run_date}] updated price history for {stock}")
    return fundamentals

@task
def enrich(fundamentals, run_date: date):
    from archive.feature_enricher import enrich_features
    # enrich_features expects `current_date`, not `as_of`
    return enrich_features(fundamentals, current_date=run_date)

@task
def run_filter(run_date: date):
    from models.run_stock_filter import run_stock_filter
    run_stock_filter(as_of=run_date)
    from core.data_provider.data_provider import load_data
    return load_data(settings.ml_selected_stocks_table)

@task
def backtest_and_label(selected, run_date: date):
    from core.backtest_bt import run_backtest
    from agents.memory.feedback_loop import update_training_data
    logger = get_run_logger()
    for rec in selected.to_dict(orient="records"):
        run_backtest(**rec, run_date=run_date)
        logger.info(f"[{run_date}] backtested {rec['stock']}")
    update_training_data()
    logger.info(f"[{run_date}] feedback loop complete")

@task
def check_drift_and_trigger(run_date: date):
    from agents.memory.memory_agent import MemoryAgent
    ma = MemoryAgent()
    ma.check_retraining_needed(as_of=run_date)

@task
def retrain_models(run_date: date):
    from models.train_stock_filter_model import train_stock_filter_model
    from models.train_dual_model_sql import train_dual_model
    from models.meta_strategy_selector import train_meta_model
    log = get_run_logger()
    train_stock_filter_model(); log.info(f"[{run_date}] filter model retrained")
    train_dual_model();          log.info(f"[{run_date}] dual models retrained")
    train_meta_model();          log.info(f"[{run_date}] meta model retrained")

@flow(name=PIPELINE_NAME)
def self_learning_pipeline(run_date: date):
    session = get_session()
    # determine date to process
    last = get_last_date(session)
    target = run_date if not last else last + timedelta(days=1)
    # run
    fund = ingest_data(target)
    feat = enrich(fund, target)
    sel  = run_filter(target)
    backtest_and_label(sel, target)
    check_drift_and_trigger(target)
    retrain_models(target)
    # update metadata
    update_last_date(session, target)

if __name__ == "__main__":
     import argparse
     from datetime import date

     parser = argparse.ArgumentParser(
         description="Run the self-learning pipeline for a given date"
     )
     parser.add_argument(
         "--run-date",
         required=True,
         help="Date to process (YYYY-MM-DD)",
         type=lambda s: date.fromisoformat(s),
     )
     args = parser.parse_args()

     # Fire off the flow for that single date
     self_learning_pipeline(args.run_date)