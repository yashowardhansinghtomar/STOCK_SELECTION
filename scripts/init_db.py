# scripts/init_db.py

from db.models import Base
from db.db import engine

if __name__ == "__main__":
    print("🔧 Creating all ORM tables if not present...")
    Base.metadata.create_all(bind=engine)
    print("✅ Done.")
