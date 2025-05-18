import logging
import sys
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"planner_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class SafeFormatter(logging.Formatter):
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            record.msg = record.msg.encode("ascii", errors="ignore").decode()
            return super().format(record)

logger = logging.getLogger("TradingBot")
logger.setLevel(logging.DEBUG)

# Console handler (without emojis if not UTF-8 compatible)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(SafeFormatter("%(asctime)s | %(levelname)s | %(message)s"))

# File handler (write all logs to file safely)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Shortcuts for emoji-based logging
def success(msg): logger.info(f"✅ {msg}")
def warn(msg): logger.warning(f"⚠️ {msg}")
def error(msg): logger.error(f"❌ {msg}")
def start(msg): logger.info(f"🚀 {msg}")

logger.success = success
logger.warn = warn
logger.start = start
