# diagnosis/migrate_paths_to_sql.py
"""
Find-and-replace PATHS[...] usage to convert file-based I/O to SQL-backed I/O.

✅ Converts:
  - pd.read_csv(PATHS["key"]) → load_data("key" )
  - df.to_csv(PATHS["key"]) → save_data(df, "key"  )

🛑 Only affects keys listed in MIGRATED_TO_SQL
"""

import os
import re
from pathlib import Path

# SQL-migrated tables (used as keys in PATHS)
MIGRATED_TO_SQL = {
    "recommendations",
    "open_positions",
    "paper_trades",
    "training_data",
    "stock_labels",
    "ml_selected_stocks",
    "backtest_summaries",
    "walkforward_log",
}

ROOT_DIR = Path(".")
TARGET_EXTENSIONS = {".py"}

read_csv_pattern = re.compile(r'pd\.read_csv\s*\(\s*PATHS\["(\w+)"\]\s*\)')
to_csv_pattern = re.compile(r'(\w+)\.to_csv\s*\(\s*PATHS\["(\w+)"\](?:,\s*index=False)?\s*\)')

def transform_code(code, filepath):
    original = code
    code = read_csv_pattern.sub(lambda m: (
        f'load_data("{m[1]}" )' if m[1] in MIGRATED_TO_SQL else m.group(0)
    ), code)

    code = to_csv_pattern.sub(lambda m: (
        f'save_data({m[1]}, "{m[2]}"  )' if m[2] in MIGRATED_TO_SQL else m.group(0)
    ), code)

    return code if code != original else None

def update_file(filepath: Path, dry_run=False):
    code = filepath.read_text(encoding="utf-8")
    updated_code = transform_code(code, filepath)
    if updated_code:
        print(f"✅ Updating: {filepath}")
        if not dry_run:
            filepath.write_text(updated_code, encoding="utf-8")

def run_all(dry_run=False):
    print(f"{'🔍 Dry run' if dry_run else '🛠️ Applying changes'} — Scanning *.py files under project root...\n")
    for file in ROOT_DIR.rglob("*"):
        if file.suffix in TARGET_EXTENSIONS and file.is_file():
            update_file(file, dry_run=dry_run)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply changes instead of dry run")
    args = parser.parse_args()

    run_all(dry_run=not args.apply)
