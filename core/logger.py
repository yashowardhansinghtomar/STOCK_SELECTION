import logging
import sys
import os
from datetime import datetime
from core.config import settings

# Setup log directory
log_dir = settings.log_dir
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"planner_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Safe formatter to prevent Unicode issues
class SafeFormatter(logging.Formatter):
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            record.msg = record.msg.encode("ascii", errors="ignore").decode()
            return super().format(record)

# Initialize logger
logger = logging.getLogger(settings.logger_name)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, settings.console_log_level.upper(), logging.INFO))
console_handler.setFormatter(SafeFormatter(settings.log_format))

# File handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(getattr(logging, settings.file_log_level.upper(), logging.DEBUG))
file_handler.setFormatter(logging.Formatter(settings.log_format))

# Attach handlers
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Use a separate reference
log = logging.getLogger("rich")

def success(msg): log.info(f"✅ {msg}")
def warn(msg): log.warning(f"⚠️ {msg}")
def error(msg): log.error(f"❌ {msg}")
def start(msg): log.info(f"🚀 {msg}")

logger.success = success
logger.warn = warn
logger.error = error
logger.start = start
