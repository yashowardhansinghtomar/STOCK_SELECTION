# core/data_provider/symbols.py

from db.postgres_manager import get_all_symbols
from core.skiplist.skiplist import get_skiplist
from core.logger.logger import logger


def get_usable_symbols() -> list:
    all_symbols = get_all_symbols()
    skiplist = set(get_skiplist())
    usable = [s for s in all_symbols if s not in skiplist]

    if skiplist:
        logger.info(f"⏭️ Skipped {len(skiplist)} skiplist stocks — using {len(usable)} usable symbols.")
    else:
        logger.info(f"✅ No stocks in skiplist. Using all {len(usable)} symbols.")

    return usable
