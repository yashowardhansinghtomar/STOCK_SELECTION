## 0️⃣ - Where we stand **this minute**

| Area                                 | Implementation status                           | Proof in repo                                                       |
| ------------------------------------ | ----------------------------------------------- | ------------------------------------------------------------------- |
| **DuckDB feature store**             | ✅ Live – cache-on-miss, primary-key indexed     | `core/feature_store/feature_store.py`                               |
| **Event bus** (Redis Streams)        | ✅ Publishing & subscribe helpers                | `core/event_bus.py`                                                 |
| **Joint Policy** (LightGBM)          | ✅ Model class + trainer + shadow inference      | `models/joint_policy.py`, `training/train_joint_policy.py`          |
| **RL stack** (Gym env + PPO trainer) | ✅ Training + replay summary log flows connected | `core/rl/gym_env.py`, `core/rl/ppo_trainer.py`, `flows/train_rl.py` |
| **Policy routing**                   | ✅ RL/Joint/Mix supported via config             | `core/policy/__init__.py`, `core/system_state.py`                   |
| **Sharpe-based allocation gate**     | ✅ Daily cron deployed via Prefect               | `core/predict/policy_chooser.py`, `policy_chooser_deploy.yaml`      |
| **Replay enrichment**                | ✅ Missed-trade, rejected-trade, holding-penalty | `agents/signal_arbitration_agent.py`, `agents/replay_logger.py`     |
| **Skiplist TTL**                     | ✅ Schema & cleanup flow completed               | `core/skiplist/skiplist.py`, `flows/clean_skiplist_flow.py`         |
| **Bootstrap simulation**             | ✅ Multi-phase runner with reward logging        | `bootstrap/run_full_bootstrap.py`, `bootstrap/bootstrap_trader.py`  |

---

## 1️⃣ 🔧  **Top three jobs now**

| #     | Change                                                                                 | Why first?                                | Rough lift                                          |
| ----- | -------------------------------------------------------------------------------------- | ----------------------------------------- | --------------------------------------------------- |
| **1** | ✅ **Instrument richer rewards**<br>holding-cost, slippage & regime tags in replay rows | RL needs realistic credit signals         | Done: `replay_logger.py`, `gym_env.py`              |
| **2** | ✅ **Replay summary on cron + Sharpe plots**                                            | Daily training snapshot + visual feedback | `track_replay_summary_flow.py` ready; cron optional |
| **3** | ✅ **Bootstrap data generation phase 0–2**                                              | Unblocks training for all policies        | `run_full_bootstrap.py` and supporting flows        |

---

## 2️⃣ 🗺️  **Migration path – week-by-week**

| Week   | Goal                                                                        | New code / config                                            | Co-existence strategy                                     |
| ------ | --------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| **W1** | ✅ Reward instrumentation<br>✅ Skiplist TTL<br>✅ `policy_chooser` scheduling | `policy_chooser_deploy.yaml`, `track_replay_summary_flow.py` | Grid/Joint/RL run side-by-side; allocation adjusts daily. |
| **W2** | ✅ **Bootstrap system (phase 0–2)**                                          | `bootstrap/run_full_bootstrap.py`, `bootstrap_trader.py`     | Uses past OHLCV to simulate training data                 |
| **W3** | **Shadow RL inference live** (10 % allocation cap)                          | Set `rl_allocation = 10` in system config                    | `ExecutionAgentSQL` uses chosen policy type.              |
| **W4** | **Joint + RL model distillation loop**                                      | Activate `flows/train_joint_policy_from_rl.py` job           | Joint still live; RL trained from high-confidence trades. |
| **W5** | **Retire Grid & Param fallbacks** in `StrategyAgent`                        | Disable grid search paths, simplify planning logic           | StrategyAgent chooses policy from system config.          |
| **W6** | **Parallel RL training** (vector envs or JAX SB3)                           | Use `gym.vector.SyncVectorEnv`, run multi-actor setup        | Same replay buffer, faster transitions per day.           |

---

## 3️⃣ 🧬  **Joint Policy evolution blueprint**

1. **Model bump**: swap LightGBM for Tiny Transformer (e.g., 4 layers × 128 dim) with `enter`, `size`, `exit` heads.
2. **Training**:

   * *Phase A:* ✅ Clone RF/Grid trades — done.
   * *Phase B:* ⏳ Daily distillation from top-RL checkpoints — to begin Sprint 2.
3. **Serving**: TorchScript export + reload via unified `core.policy.predict()`.
4. **Retire LightGBM**: once AUC + latency thresholds are beat.

---

## 4️⃣ ⚙️  **Feature-flow verdict**

🟢 **Stick with DuckDB** through 2025.

* Already in use by RL env, feature backfill, and Joint trainer.
* Fast, embedded, cheap to maintain.
* Next: add weekly compaction job to control DB size (< 2 GB).

Kafka or Redis pub/sub can come later for remote inference or async UI triggers.

---

## 5️⃣ 🤖  **Concrete checklist to make RL “captain of the ship”**

| Component             | Required change                                                                               |
| --------------------- | --------------------------------------------------------------------------------------------- |
| **ExecutionAgentSQL** | ✅ Publish `FILL`, `M2M_PNL`, `UNWIND` to Redis stream.                                        |
| **Replay buffer**     | ✅ Include `holding_days`, `regime_tag`, `slippage`.                                           |
| **Gym env**           | ✅ Use these to compute `reward = pnl - hold_penalty - slippage_penalty + capital_efficiency`  |
| **Trainer**           | ⏳ Try AWR (advantage-weighted regression) or AWAC to stabilize continuous `size` head.        |
| **Inference path**    | ✅ `core.policy.predict()` routes to RL / Joint / Mix cleanly.                                 |
| **Data volume**       | ✅ Bootstrap + replay logs enable >200 transitions/day, scaling to 20k+/week with vector envs. |

---

## 6️⃣ 📅  **At-a-glance sprint board**

| Sprint (2-week cadence)                  | Epic                          | Key deliverables                                                    |
| ---------------------------------------- | ----------------------------- | ------------------------------------------------------------------- |
| **Sprint 1** (✅ complete)                | *Reward + skiplist + chooser* | • Holding/slip rewards live<br>• Skiplist TTL<br>• Sharpe dashboard |
| **Sprint 2** (▶ now live)                | *Bootstrap + shadow trading*  | • Phase 0–2 bootstrap done<br>• RL gets 10 % capital                |
| **Sprint 3** (Jun 29 → Jul 12)           | *Model consolidation*         | • Remove Grid/Param<br>• Promote Joint+RL only                      |
| **Sprint 4** (Jul 13 → Jul 26)           | *Parallel RL & regime tags*   | • Vectorised PPO<br>• Volatility regime tagging in replay           |
| **Sprint 5** (Jul 27 → Aug 9)            | *RL majority control*         | • Allocation gate ≥ 70 %<br>• LGBM → Tiny-Transformer prototype     |
| **Hardening / buffer** (Aug 10 → Aug 23) | *Clean-up & docs*             | • Delete dead tables/flows<br>• Full pipeline runbook               |

---

### ✔️ Immediate next actions

1. 🔛 Phase 0–2 already running via `run_full_bootstrap.py`.
2. ✅ Replay enriched with synthetic + guided trades.
3. 🔜 Set `rl_allocation = 10` to let RL infer on today's picks.
4. 🔜 Begin off-policy distillation job for JointPolicy from RL replay.
