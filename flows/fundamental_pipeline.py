from pathlib import Path
import os
import time

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from prefect import flow, task, get_run_logger
from core.logger import logger
from core.data_provider import load_data, save_data
from core.skiplist import is_in_skiplist, add_to_skiplist
from db.postgres_manager import run_query

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
CACHE_DIR    = Path("cache/fundamentals")
INPUT_CSV    = Path("fundamentals/EQUITY.csv")
TABLE_NAME   = "stock_fundamentals"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_nse_symbols():
    df = pd.read_csv(INPUT_CSV)
    return df[["SYMBOL", "NAME_OF_COMPANY"]].dropna()


def is_cache_valid(sym: str, max_age_days: int = 7) -> bool:
    p = CACHE_DIR / f"{sym}.json"
    if not p.exists(): 
        return False
    return (time.time() - p.stat().st_mtime) < max_age_days * 86400


def save_local_cache(sym: str, data: dict):
    (CACHE_DIR / f"{sym}.json").write_text(pd.json.dumps(data))


def clear_sql_table():
    try:
        run_query(f'DELETE FROM "{TABLE_NAME}"', fetchall=False)
        logger.info(f"🗑️ Cleared SQL table {TABLE_NAME}")
    except Exception as e:
        logger.error(f"❌ Could not clear SQL table {TABLE_NAME}: {e}")


def clear_local_cache():
    removed = 0
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()
        removed += 1
    logger.info(f"🗑️ Cleared {removed} cached JSON files")


# ────────────────────────────────────────────────────────────────────────────────
# Fetch & parse
# ────────────────────────────────────────────────────────────────────────────────
def _scrape_screener(symbol):
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    ratios = {}
    for row in soup.select("table.ratios-table tr"):
        cells = row.find_all("td")
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True)
            val = cells[1].get_text(strip=True).replace(",", "")
            ratios[key] = val
    return {
        "trailingPE":     float(ratios.get("P/E", "0") or 0),
        "debtToEquity":   float(ratios.get("Debt to equity", "0") or 0),
        "returnOnEquity": float(ratios.get("ROE %", "0") or 0) / 100,
        "marketCap":      None,
        "sector":         "Unknown",
        "industry":       "Unknown",
    }


def _fetch_yfinance(symbol):
    t = yf.Ticker(f"{symbol}.NS")
    info = t.info or {}
    if "symbol" not in info:
        raise KeyError("yfinance returned no info")
    return info


def fetch_fundamentals(symbol: str) -> dict:
    # 1) Try Screener.in
    try:
        return _scrape_screener(symbol)
    except Exception:
        pass

    # 2) Fallback to NSE India API
    try:
        sess = requests.Session()
        sess.get("https://www.nseindia.com", timeout=5)
        data = sess.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={symbol}",
            timeout=10
        ).json().get("info", {})
        return {
            "trailingPE":   float(data.get("pe", 0) or 0),
            "debtToEquity": None,
            "returnOnEquity": None,
            "marketCap":    float(data.get("marketCap", 0) or 0),
            "sector":       "Unknown",
            "industry":     "Unknown",
        }
    except Exception:
        pass

    # 3) Finally yfinance
    return _fetch_yfinance(symbol)


def parse_fundamentals(symbol: str, name: str, info: dict) -> dict:
    return {
        "stock":           symbol,
        "name":            name,
        "pe_ratio":        info.get("trailingPE"),
        "debt_to_equity":  info.get("debtToEquity"),
        "roe":             (info.get("returnOnEquity") or 0) * 100,
        "earnings_growth": info.get("earningsGrowth"),
        "market_cap":      info.get("marketCap"),
        "sector":          info.get("sector", "Unknown"),
        "industry":        info.get("industry", "Unknown"),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Prefect Tasks
# ────────────────────────────────────────────────────────────────────────────────
@task
def clear_everything(force: bool):
    if force:
        clear_sql_table()
        clear_local_cache()


@task
def get_todo_symbols():
    all_syms = load_nse_symbols()
    done     = load_data(TABLE_NAME).get("stock", []).tolist() if not load_data(TABLE_NAME).empty else []
    return [
        (row.SYMBOL, row.NAME_OF_COMPANY)
        for _, row in all_syms.iterrows()
        if row.SYMBOL not in done and not is_in_skiplist(row.SYMBOL)
    ]


@task(retries=0)
def fetch_one(symbol_name):
    symbol, name = symbol_name
    log = get_run_logger()

    # local cache hit?
    if is_cache_valid(symbol):
        raw = pd.json.loads((CACHE_DIR / f"{symbol}.json").read_text())
        return parse_fundamentals(symbol, name, raw)

    try:
        raw = fetch_fundamentals(symbol)
        save_local_cache(symbol, raw)
        return parse_fundamentals(symbol, name, raw)
    except KeyError as ke:
        # permanent no‐data → skip forever
        add_to_skiplist(symbol, str(ke))
        log.warning(f"Skipping {symbol} permanently: {ke}")
        return None
    except Exception as e:
        # transient or rate-limit → let Prefect retry next run
        log.error(f"Transient error for {symbol}: {e}")
        return None


@task
def save_batch(rows):
    clean = [r for r in rows if r]
    if not clean:
        return 0
    df = pd.DataFrame(clean)
    # volume proxy filter
    df["volume_proxy"] = df["market_cap"] / df["pe_ratio"].replace(0, np.nan)
    df = df[df["volume_proxy"] > 1e6]
    save_data(df, TABLE_NAME, if_exists="append")
    return len(df)


# ────────────────────────────────────────────────────────────────────────────────
# Prefect Flow
# ────────────────────────────────────────────────────────────────────────────────
@flow(name="Fundamentals Fetcher")
def fundamental_fetch_flow(force: bool = False):
    """
    1) Optionally clear out everything if --force
    2) Figure out which symbols still need data
    3) Fan out one task per symbol (up to your concurrency limit)
    4) Persist the successful rows back into SQL
    5) On subsequent Prefect runs you’ll only fetch the delta
       until “get_todo_symbols” returns empty → you’re done.
    """
    clear_everything(force)
    todo = get_todo_symbols()

    if not todo:
        logger.info("✅ All fundamentals fetched — nothing to do.")
        return

    results = fetch_one.map(todo)
    count   = save_batch(results)

    logger.info(f"Fetched & saved {count}/{len(todo)} new rows this run; {len(todo)-count} remaining.")


if __name__ == "__main__":
    # run with `python -m fundamentals.prefect_fundamental_flow --force`
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="wipe & re-fetch everything")
    args = p.parse_args()
    fundamental_fetch_flow(force=args.force)
