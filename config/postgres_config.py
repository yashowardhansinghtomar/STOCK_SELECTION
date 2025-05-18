"""
Provides PostgreSQL connection credentials and helpers.

Usage:
from config.postgres_config import get_pg_conn_params
"""

import os

def get_pg_conn_params():
    return {
        "host": os.getenv("PG_HOST", "localhost"),
        "port": int(os.getenv("PG_PORT", "5432")),
        "user": os.getenv("PG_USER", "postgres"),
        "password": os.getenv("PG_PASSWORD", "0809"),
        "dbname": os.getenv("PG_DB", "trading_db")
    }
# postgrest.conf
DB_URI = "postgres://postgres:0809@localhost:5432/trading_db"
DB_SCHEMA = "public"
DB_ANON_ROLE = "anon"
SERVER_PORT = 3000
