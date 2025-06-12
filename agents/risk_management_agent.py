# agents/risk_management_agent.py

from core.logger.logger import logger
from agents.replay_logger import log_replay_row

class RiskManagementAgent:
    def __init__(self):
        pass  # Add init config if needed later

    def approve(self, signal: dict) -> bool:
        """
        Evaluate the signal against risk rules.
        Returns True if trade is allowed; False otherwise.
        """
        # Example risk rules (add more as needed)
        if signal.get("confidence", 0) < 0.3:
            self._log_reject(signal, reason="low_confidence")
            return False

        if signal.get("max_drawdown", 0) > 0.25:
            self._log_reject(signal, reason="high_drawdown")
            return False

        # Accept by default
        return True

    def _log_reject(self, signal: dict, reason: str):
        logger.info(f"⛔ Rejected {signal['stock']} due to {reason}")
        log_replay_row(
            stock=signal["stock"],
            action="none",
            reason=reason,
            model=signal.get("model", "unknown"),
            prediction=signal.get("predicted_return"),
            confidence=signal.get("confidence"),
            signal=signal.get("signal", "unknown"),
            date=signal.get("date")
        )
