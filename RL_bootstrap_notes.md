🧠 Dual Pass RL Strategy (for Bootstrap Script)
Purpose: Combine exploration (for learning) and exploitation (for evaluation).

Steps:

Pass 1 – Exploration Mode
Use predict_action(..., deterministic=False)
→ Collect diverse replay episodes for training/fine-tuning

Pass 2 – Exploitation Mode
Use predict_action(..., deterministic=True)
→ Evaluate model performance using stable policy output

Benefits:

Prevents overfitting to early deterministic patterns

Builds richer replay buffer

Improves generalization before deployment

To-do: Add toggle or config in bootstrap controller to switch between modes per pass.