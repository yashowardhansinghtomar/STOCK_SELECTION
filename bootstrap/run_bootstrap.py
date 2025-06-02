# run_bootstrap.py

import argparse
from bootstrap.historical_bootstrap_runner import run_historical_bootstrap


def main():
    parser = argparse.ArgumentParser(description="Run Historical Bootstrap Simulation")
    parser.add_argument("--start", type=str, required=True, help="Start date in YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date in YYYY-MM-DD")

    args = parser.parse_args()
    run_historical_bootstrap(args.start, args.end)


if __name__ == "__main__":
    main()
