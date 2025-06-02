import logging
import sys
import os
import re
from datetime import datetime
from core.config.config import settings

# Setup log directory
log_dir = settings.log_dir
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"planner_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# === Safe Unicode Filtering ===
def strip_surrogates(text):
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'[\ud800-\udfff]', '', text)

class SafeFormatter(logging.Formatter):
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            record.msg = strip_surrogates(record.msg)
            return super().format(record)

# Core logger
logger = logging.getLogger(settings.logger_name)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

# Console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, settings.console_log_level.upper(), logging.INFO))
console_handler.setFormatter(SafeFormatter(settings.log_format))

# File output
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(getattr(logging, settings.file_log_level.upper(), logging.DEBUG))
file_handler.setFormatter(SafeFormatter(settings.log_format))

# Attach handlers
logger.handlers = []
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# === Logging helpers — Unicode-safe with emoji support ===
def success(msg, prefix=""):
    logger.info(f"{prefix}✅ {strip_surrogates(msg)}")

def warnings(msg, prefix=""):
    logger.warning(f"{prefix}⚠️ {strip_surrogates(msg)}")

def errors(msg, prefix=""):
    logger.error(f"{prefix}❌ {strip_surrogates(msg)}")

def start(msg, prefix=""):
    logger.info(f"{prefix}🚀 {strip_surrogates(msg)}")

# Attach custom methods
logger.success = success
logger.warnings = warnings
logger.errors = errors
logger.start = start
