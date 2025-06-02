# 🧠 O.D.I.N. — Next Horizon: Intelligent Autonomy

This document tracks the progress of O.D.I.N.’s upcoming leap: evolving from clever automation into a **regime-aware, capital-efficient, continuously-learning trading intelligence**.

---

## 1 🚀 Next Strategic Moves (high-impact upgrades)

| Priority | Change                                              | Why it matters                                                                                   |
| -------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **1**    | **Regime-aware policy switching**                   | RL models should act differently in high-volatility vs trending markets.                         |
| **2**    | **Capital-weighted reward functions**               | Adds capital efficiency, holding cost, and slippage into RL learning.                            |
| **3**    | **Position sizing policy head**                     | Move from binary “enter” to continuous “how much?” trades for better capital allocation.         |

---

## 2 🧠 Design Proposal — “Contextual & Capital-Aware PPO”

| Component                     | Description                                                                                          |
|------------------------------|------------------------------------------------------------------------------------------------------|
| **Input features**           | OHLCV × N bars + technical indicators + market regime tags (volatility, macro context, etc.)         |
| **Architecture**             | JointPolicy base + new `context_vector` (regime, sentiment) + multi-head outputs                     |
| **Outputs**                  | `enter` (sigmoid), `position_size` (softplus), `exit_rule` (categorical)                            |
| **Reward design**            | `+ PnL - holding_penalty - slippage_penalty + capital_efficiency_bonus`                             |
| **Replay tags**              | Each transition tagged with regime context + margin used                                             |

---

## 3 ⚙️ Implementation Blocks

| Block                                 | Status     | Objective                                                                 |
|--------------------------------------|------------|--------------------------------------------------------------------------|
| **1. Regime labelling engine**       | 🕐 Planned | Add VIX-levels / trend-strength / volume-volatility scores to each bar   |
| **2. RL reward expansion**           | 🕐 Planned | Modify gym_env to support extended rewards (PnL + penalties)             |
| **3. Position-size head**            | 🕐 Planned | Add softplus head for `size`; train with replay-calibrated reward        |
| **4. Vectorized training**           | 🕐 Planned | Speed up learning via gymnasium + JAX / SB3 parallel envs                |
| **5. Curriculum bootstrapping**      | 🕐 Planned | Use easier windows first → gradually include harder market regimes       |
| **6. Policy distillation (edge)**    | 🕐 Planned | Export PPO policy to 2-layer MLP for bar-wise inference without latency  |
| **7. Unified dashboard**             | ⏳ Drafting | Track RL Sharpe, drawdown, feature freshness, slippage vs predicted      |

---

## 4 🔁 Feedback & Learning Signals Plan

| Signal                                         | Purpose                                                              | Captured via                       |
|-----------------------------------------------|----------------------------------------------------------------------|------------------------------------|
| **Missed-trade virtual P&L**                   | Helps RL learn from *inaction*                                       | Already partially done             |
| **Holding-period penalty**                    | Penalizes long, dead trades                                          | Extend reward in replay logger     |
| **Slippage vs prediction**                    | Penalizes models that over-promise                                   | Compare predicted return vs actual |
| **Capital efficiency score**                  | Rewards trades that return more per unit capital                     | Use margin usage estimate          |
| **Regime tag + context state**                | Enables switching behavior between bullish/bearish regimes           | Tag replay + model input           |

---

## 5 📅 Phase Plan — Q3 2025 Milestones

| Month        | Milestone                                                           | Success Metric                                               |
|--------------|---------------------------------------------------------------------|--------------------------------------------------------------|
| **June**     | Regime labels, reward function refactor                             | New rewards show up in replay rows                           |
| **July**     | PPO model with size-head + regime inputs + retrain on replay        | Model uses context & scales size effectively                 |
| **Aug**      | Curriculum strategy, dashboard logging, PPO distillation prototype  | Model runs bar-wise in <5ms, stream dashboard goes live      |

---

## 💡 Closing Thought

The automation layer is stable. This next horizon gives O.D.I.N. the **agency to adapt, optimize, and evolve** — not just execute what it’s told.

*Once O.D.I.N. senses regime shifts and adjusts both *whether* and *how much* to trade, we’ll move from a smart script to a true AI trader.*
