# db/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.config.config import settings

# Initialize SQLAlchemy engine and session factory using env-configured URL
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_session():
    """Returns a new SQLAlchemy Session."""
    return SessionLocal()
