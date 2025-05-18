from sqlalchemy import inspect, create_engine
from db.models import Base

# 🔧 Hardcoded correct connection string for testing
db_url = "postgresql://postgres:0809@localhost:5432/trading_db"
print(db_url)

engine = create_engine(db_url)

def check_schema():
    inspector = inspect(engine)
    for table_class in Base.__subclasses__():
        table_name = table_class.__tablename__
        if not inspector.has_table(table_name):
            print(f"❌ Table missing: {table_name}")
            continue
        db_cols = {col["name"] for col in inspector.get_columns(table_name)}
        orm_cols = set(table_class.__table__.columns.keys())
        missing = orm_cols - db_cols
        extra = db_cols - orm_cols
        if missing or extra:
            print(f"⚠️  {table_name}")
            if missing:
                print(f"   - Missing in DB: {sorted(missing)}")
            if extra:
                print(f"   - Extra in DB:   {sorted(extra)}")
        else:
            print(f"✅ {table_name} is in sync.")

if __name__ == "__main__":
    check_schema()

