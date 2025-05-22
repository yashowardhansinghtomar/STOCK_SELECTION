# agents/signal_arbitration_agent.py

from core.logger import logger
import numpy as np


class SignalArbitrationAgent:
    def __init__(self):
        pass

    def arbitrate(self, signals: list):
        if not signals:
            logger.warning("No signals provided for arbitration.")
            return None

        weighted_scores = {
            'entry_exit_model': 0.6,
            'ml': 0.5,
            'grid_fallback': 0.3,
            'ts_fallback': 0.2,
            'rl_agent': 0.4,
        }

        signals = sorted(signals, key=lambda s: weighted_scores.get(s['source'], 0), reverse=True)

        final_signal = signals[0]
        logger.info(f"🎯 Final arbitrated signal: {final_signal['source']} for {final_signal['stock']}")
        return final_signal
