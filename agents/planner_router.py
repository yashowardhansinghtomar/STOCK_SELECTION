# agents/planner_router.py

from agents.planner_agent_sql import PlannerAgentSQL
from agents.intraday_planner_agent import IntradayPlannerAgent
from core.logger import logger

def run_all_planners():
    logger.start("🔁 Running ALL strategy planners...")

    try:
        logger.info("📆 Executing daily PlannerAgentSQL...")
        PlannerAgentSQL().run()
    except Exception as e:
        logger.error(f"❌ Daily Planner failed: {e}")

    try:
        logger.info("⏱️ Executing intraday IntradayPlannerAgent...")
        IntradayPlannerAgent().run()
    except Exception as e:
        logger.error(f"❌ Intraday Planner failed: {e}")

    logger.success("✅ All strategy planners completed.")


if __name__ == "__main__":
    run_all_planners()
