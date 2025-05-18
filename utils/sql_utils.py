# utils/sql_utils.py
from config.sql_tables import SQL_TABLES

def is_sql_table(key: str) -> bool:
    return key in SQL_TABLES
