# db/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config.postgres_config import get_pg_conn_params
from core.config import settings 

# Load Postgres connection parameters
params = get_pg_conn_params()
DB_URL = (
    f"postgresql+psycopg2://{params['user']}:{params['password']}@"
    f"{params['host']}:{params['port']}/{params['dbname']}"
)

# Initialize SQLAlchemy engine and session factory
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    """
    Returns a new SQLAlchemy Session.
    Usage:
        with get_session() as session:
            ...
    """
    return SessionLocal()