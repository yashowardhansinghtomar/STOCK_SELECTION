# regime_meta_controller.py

class RegimeAdaptiveMetaController:
    def __init__(self):
        self.epsilon = 0.9
        self.position_size = 0.001

    def update(self, vol_regime, progress=0.0):
        """
        Adjust epsilon and position size based on volatility regime and learning progress [0, 1]
        """
        if vol_regime == "low":
            self.epsilon = 0.4 + 0.5 * (1 - progress)
            self.position_size = 0.003
        elif vol_regime == "med":
            self.epsilon = 0.2 + 0.3 * (1 - progress)
            self.position_size = 0.002
        elif vol_regime == "high":
            self.epsilon = 0.05 + 0.1 * (1 - progress)
            self.position_size = 0.001

    def get_epsilon(self):
        return round(self.epsilon, 3)

    def get_position_size(self):
        return round(self.position_size, 4)
