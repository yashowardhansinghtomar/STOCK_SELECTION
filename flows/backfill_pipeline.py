# flows/backfill_pipeline.py
from datetime import date, timedelta
from prefect import flow

from flows.auto_pipeline import self_learning_pipeline

@flow(name="historical-backfill")
def historical_backfill(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        self_learning_pipeline(current)
        current += timedelta(days=1)

if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(
        description="Run historical backfill for the self-learning pipeline"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="First date to backfill (YYYY-MM-DD)",
        type=lambda s: date.fromisoformat(s),
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Last date to backfill (YYYY-MM-DD)",
        type=lambda s: date.fromisoformat(s),
    )
    args = parser.parse_args()

    # Execute the flow for each day in the range
    historical_backfill(args.start_date, args.end_date)