# bootstrap/replay_buffer.py

from collections import defaultdict
from statistics import mean

REAL_SOURCES = {"guided", "autonomous", "real"}  # all acceptable source tags

class ReplayBuffer:
    def __init__(self, tag="bootstrap"):
        self.data = []
        self.tag = tag
        self.metrics = defaultdict(list)

    def add(self, trade, tags=None):
        if tags is None:
            tags = {}
        trade_entry = {
            **trade,
            "tags": tags,
        }
        self.data.append(trade_entry)
        self._track_metrics(trade_entry)

    def _track_metrics(self, trade_entry):
        phase = trade_entry["tags"].get("phase")
        reward = trade_entry.get("reward", 0)
        regime = trade_entry["tags"].get("vol_regime", "unknown")
        self.metrics[f"reward_phase_{phase}"].append(reward)
        self.metrics[f"reward_regime_{regime}"].append(reward)

    def count_real_trades(self):
        return len([t for t in self.data if t["tags"].get("source") in ("guided", "autonomous")])

    def policy_converged(self):
        recent_rewards = self.metrics.get("reward_phase_1", [])[-50:]
        return len(recent_rewards) >= 50 and abs(mean(recent_rewards)) > 0.1

    def sharpe(self, regime):
        rewards = self.metrics.get(f"reward_regime_{regime}", [])
        if not rewards or len(rewards) < 2:
            return 0.0
        avg = mean(rewards)
        std = (sum((r - avg)**2 for r in rewards) / (len(rewards) - 1))**0.5
        return round(avg / std, 2) if std != 0 else 0.0

    def trade_coverage(self, symbol_list):
        traded_symbols = set(t["symbol"] for t in self.data)
        covered = [s for s in symbol_list if s in traded_symbols]
        return round(100 * len(covered) / len(symbol_list), 2) if symbol_list else 0.0

    def count_trades(self, regime):
        return len([t for t in self.data if t["tags"].get("vol_regime") == regime])


    def count_real_trades(self):
        return len([t for t in self.data if t["tags"].get("source") in REAL_SOURCES])

