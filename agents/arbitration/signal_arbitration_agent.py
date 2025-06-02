from core.logger.logger import logger
from agents.replay_logger import log_replay_row
from predict.ppo_live_policy import PPOLivePolicy

class SignalArbitrationAgent:
    def __init__(self):
        self.weighted_scores = {
            'ppo_sb3': 0.85,
            'joint_policy': 0.8,
            'entry_exit_model': 0.6,
            'ml': 0.5,
            'grid_fallback': 0.3,
            'ts_fallback': 0.2,
            'rl_agent': 0.4,
        }
        self.ppo_policy = PPOLivePolicy()

    def arbitrate(self, signals: list, log: bool = True):
        if not signals:
            logger.warning("[ARBITRATION] No signals provided.")
            return None

        stock = signals[0].get("stock")
        date = signals[0].get("date")
        has_ppo = any(sig.get("source") == "ppo_sb3" for sig in signals)

        if not has_ppo:
            ppo_signal = self.ppo_policy.predict(stock, date)
            if ppo_signal:
                signals.append(ppo_signal)
                logger.info(f"[ARBITRATION] Injected PPO signal for {stock}")

        def score(signal):
            return self.weighted_scores.get(signal.get('source', ''), 0)

        # Compute and attach scores
        for s in signals:
            s["arbitration_score"] = score(s)

        signals = sorted(signals, key=lambda s: s["arbitration_score"], reverse=True)
        final_signal = signals[0]

        if log:
            logger.info(f"⚖️ Arbitration decision for {stock} on {date}")
            for sig in signals:
                log_msg = (
                    f" - {sig.get('source', 'unknown')}: score={sig['arbitration_score']:.2f}, "
                    f"signal={sig.get('signal', '?')}, "
                    f"confidence={sig.get('confidence', '?')}"
                )
                if sig == final_signal:
                    logger.info("✅ SELECTED " + log_msg)
                else:
                    logger.info("❌ DISCARDED " + log_msg)
                    log_replay_row(
                        stock=sig.get("stock"),
                        action="none",
                        reason="not_selected",
                        model=sig.get("model", sig.get("source", "unknown")),
                        prediction=sig.get("predicted_return"),
                        confidence=sig.get("confidence"),
                        signal=sig.get("signal", "unknown"),
                        date=sig.get("date")
                    )

        return final_signal
