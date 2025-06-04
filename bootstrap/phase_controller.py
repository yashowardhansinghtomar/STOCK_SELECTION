# phase_controller.py
from core.market_conditions import get_volatility_regime
from bootstrap.trade_generator import (
    generate_random_trades,
    generate_rule_based_trades,
    generate_model_based_trades
)

import random


class PhaseController:
    def __init__(self, initial_phase=0):
        self.phase = initial_phase
        self.epsilon = 0.9  # starting exploration rate

    def update_phase(self, replay_buffer):
        real_count = replay_buffer.count_real_trades()
        if self.phase == 0 and real_count > 200:
            self.phase = 1
        elif self.phase == 1 and replay_buffer.policy_converged():
            self.phase = 2

        # Adjust epsilon gradually
        self.epsilon = max(0.1, self.epsilon - 0.01)

    def generate_trades(self, filtered_stocks, date):
        vol_regime = get_volatility_regime(date)

        if self.phase == 0:
            return generate_random_trades(filtered_stocks, date, vol_regime)

        elif self.phase == 1:
            # ε-greedy: mix rule-based and random
            trades = []
            for stock in filtered_stocks:
                if random.random() < self.epsilon:
                    trades += generate_rule_based_trades([stock], date, vol_regime)
                else:
                    trades += generate_random_trades([stock], date, vol_regime)
            return trades

        else:  # phase >= 2
            return generate_model_based_trades(filtered_stocks, date, vol_regime)

    def get_source_label(self):
        return {0: "synthetic", 1: "guided", 2: "autonomous"}.get(self.phase, "autonomous")
