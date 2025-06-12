# utils/cache.py

# Global or shared context for one run
seen_skips = set()

def is_skipped(symbol: str) -> bool:
    if symbol in seen_skips:
        return True
    session = SessionLocal()
    try:
        exists = session.query(SkiplistStock).filter_by(stock=symbol).first() is not None
        if exists:
            seen_skips.add(symbol)
        return exists
    finally:
        session.close()