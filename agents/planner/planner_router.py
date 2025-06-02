# agents/planner_router.py


import threading
from agents.planner.planner_agent_sql import PlannerAgentSQL
from agents.planner.intraday_planner_agent import IntradayPlannerAgent
from core.logger.logger import logger
import time
with open("odin_banner.txt") as f:
    logger.info(f.read())

def run_daily_planner():
    try:
        logger.start("🌅 Running daily planner (PlannerAgentSQL)...")
        PlannerAgentSQL(dry_run=False).run()
        logger.success("✅ Daily planner completed.")
    except Exception as e:
        logger.error(f"❌ Daily planner failed: {e}")

def run_intraday_loop():
    try:
        logger.start("📡 Starting intraday real-time planner...")
        IntradayPlannerAgent(dry_run=False).run_forever()
    except Exception as e:
        logger.error(f"❌ Intraday loop failed: {e}")

def run_all_planners():
    run_daily_planner()

    # 🧵 Start intraday planner in background thread
    intraday_thread = threading.Thread(target=run_intraday_loop)
    intraday_thread.daemon = True
    intraday_thread.start()

    # Keep main thread alive
    while True:
        time.sleep(60)

if __name__ == "__main__":
    run_all_planners()
