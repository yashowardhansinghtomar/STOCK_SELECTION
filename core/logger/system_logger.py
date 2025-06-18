# core/logger/system_logger.py

import json
from datetime import datetime
import pandas as pd
from core.logger.logger import logger
from core.time_context.time_context import get_simulation_date
from db.postgres_manager import run_query, insert_dataframe

TABLE_NAME = "system_log"

def log_event(agent: str, module: str, action: str, result: str, meta: dict = None):
    now = pd.to_datetime(datetime.now())
    sim_date = pd.to_datetime(get_simulation_date()).date()

    row = {
        "simulation_date": sim_date,
        "timestamp": now,
        "agent": agent,
        "module": module,
        "action": action,
        "result": result,
        "meta": json.dumps(meta or {}),
    }

    try:
        insert_dataframe(pd.DataFrame([row]), table_name=TABLE_NAME)
        logger.debug(f"[SYSLOG] {agent}.{module} → {action}: {result}")
    except Exception as e:
        logger.warning(f"System log insert failed: {e}")
