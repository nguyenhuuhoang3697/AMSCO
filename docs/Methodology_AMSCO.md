# Methodology: Adaptive Multi‑Strategy Controller (AMSCO)

This section details the methodology for tuning the Adaptive Multi‑Strategy Controller (AMSCO) used for black‑box hyperparameter optimization. It is designed for direct inclusion in a master’s thesis. The implementation corresponds to `amsco_adult_v2.py`, with core components: Orchestrator, MetaController (UCB1), three Strategy Agents (Random, Bayesian/TPE, Grid/local refinement), a shared `KnowledgeHub`, and Optuna’s pruning/sampling.

## 1. Objective and Setting
- Primary objective: maximize classification accuracy; report F1 and ROC‑AUC as secondary metrics.
- Budgeted optimization: total trials `TOTAL_TRIALS` split into slices `SLICE_BUDGET` for adaptive allocation.
- Estimation regime: cross‑validation (CV) for robust inner‑loop evaluation; optional Nested‑CV for outer‑loop validation.

## 2. Architecture Summary
- Orchestrator: runs the optimization loop with `total_budget` and `slice_budget`.
- Strategy Agents: explore the shared search space via distinct policies:
  - RandomAgent: uniform/log‑uniform sampling baseline.
  - BayesianAgent: Optuna’s `TPESampler(seed=42)` with `MedianPruner`.
  - GridAgent: local exploitation around current best; bounded loops and early stop.
- KnowledgeHub: centralized store of all trials (`params`, `score`, `agent_id`, `iteration`).
- MetaController (UCB1): allocates each slice’s budget to the agent with highest UCB score, balancing exploration and exploitation.

UCB1 score for agent i at allocation step N with pulls n_i and mean reward r̄_i:

$UCB_i = \bar{r}_i + \sqrt{\frac{2\ln N}{n_i}}$.

Rewards use the CV accuracy returned by the objective (normalized implicitly to [0,1]).

## 3. Phased Tuning Protocol (G1–G8)
- G1 — Foundation: ensure consistent preprocessing (`preprocessor`), well‑typed search space (float/int/categorical; log scale where needed), and objective with CV.
- G2 — Pre‑tuning Stability: perform short runs (e.g., 15–20 trials) to assess variance; keep accuracy as reward. Enable stepwise reporting so the pruner can act within folds.
- G3 — Agent‑level Refinement:
  - RandomAgent: verify sampling honors space types and log ranges; use as broad explorer.
  - BayesianAgent: keep `TPESampler(seed=42)`, `MedianPruner(n_startup_trials=5, n_warmup_steps=0)`; allow 5–10 random warmup trials implicitly.
  - GridAgent: focus on 1–2 most influential parameters near the best known configuration; set safeguards to avoid local traps.
- G4 — Meta‑controller (UCB1): cold‑start each agent once; then allocate by $UCB_i$. Optionally add a minimal exploration floor (e.g., 5–10% slices) and use a short moving average of rewards to reduce noise.
- G5 — Estimation Strategy: use CV (k=5) during optimization for reliable signals; validate with Nested‑CV (e.g., outer=5, inner=3) for final generalization checks.
- G6 — Early‑Stopping/Resources: rely on MedianPruner’s stepwise updates; keep slices small enough (5–10) for fast feedback. Track wall/CPU time and memory; shrink the search space or reduce parallelism if needed.
- G7 — Experiments and Statistics: compare AMSCO against Random, Optuna (TPE), and Hyperopt (TPE). Report primary metric and optimization efficiency: wall time, total trials, iteration‑to‑best, and convergence ratio (below).
- G8 — Deployment & Recommendations: fix seeds, log `agent_pulls` and `agent_budget_usage`, and preserve best params and iteration‑to‑best for reproducibility. If progress stalls across 2–3 slices, tighten the space around the best and emphasize GridAgent briefly.

## 4. Default Settings (Aligned with Implementation)
- Slice budget: `SLICE_BUDGET = 5` (recommended 5–10 for responsive allocation).
- Total trials: `TOTAL_TRIALS = 50` for medium search spaces.
- Cross‑validation: `CV_FOLDS = 5` for evaluation; 3–5 for inner optimization cost control.
- BayesianAgent: `TPESampler(seed=42)`; `MedianPruner(n_startup_trials=5, n_warmup_steps=0)`.
- GridAgent safeguards: `MAX_LOOPS = 2`, `EARLY_STOP_NO_IMPROVE = 1`.
- Convergence ratio: `opt_convergence_ratio = iteration_to_best / total_trials` (lower is better).

## 5. Metrics and Reporting
- Optimization metrics: primary accuracy, secondary F1 and ROC‑AUC from CV; optional extra CV=5 diagnostics (precision, recall, balanced accuracy).
- Resource diagnostics: `opt_wall_time`, `opt_cpu_time`, `opt_peak_memory_mb`, `opt_rss_memory_mb`, `opt_total_trials`, `opt_iter_best`, and `opt_convergence_ratio`.
- Statistical comparison: paired t‑tests and Cohen’s d (Negligible/Small/Medium/Large/Very Large) for AMSCO vs baselines, on resource metrics and convergence.

## 6. Experimental Design
- Datasets: Adult (with internal preprocessing), optionally Telco and Breast Cancer.
- Models: Logistic Regression, Random Forest (optionally XGBoost, LightGBM).
- Budgets: `TOTAL_TRIALS = 50`, `SLICE_BUDGET = 5` for main runs; Nested‑CV with `outer=5`, `inner=3` using `NESTED_INNER_TRIALS` sized to match inner‑fit budget of the standard evaluation.
- Outputs: method‑wise tables of Accuracy/F1/ROC‑AUC and optimization diagnostics; summary tables for thesis: mean accuracy, std, mean time, and convergence speed (accuracy/time).

## 7. Acceptance Criteria
- Performance: AMSCO achieves accuracy comparable to or better than Optuna (±1%) while improving efficiency (lower `iteration_to_best` or `opt_convergence_ratio`).
- Stability: across three repeated runs, std(accuracy) ≤ 0.01–0.02 (absolute).
- Generalization: Nested‑CV performance within 1–2% of CV (k=5).
- Resources: memory peaks and wall/CPU time remain within practical limits under the given budget.

## 8. Reproducibility and Practical Notes
- Fix random seeds (e.g., 42) across samplers and models; keep deterministic preprocessing.
- Preserve `KnowledgeHub` logs per run; export `agent_pulls`, `agent_budget_usage`, `best_params`, and `iteration_to_best`.
- If pruning is too aggressive on very fast/noisy tasks, increase `n_startup_trials` or temporarily disable pruning for diagnostics.

## 9. Quick Run and Configuration
- Quick run (single dataset/model set as in script defaults):
  - `TOTAL_TRIALS = 50`, `SLICE_BUDGET = 5`, `CV_FOLDS = 5`, `DATASETS = ['adult']`, `MODELS = ['logistic_regression','random_forest']`.
- Execution: `python amsco_adult_v2.py` (captures resource diagnostics, optimizer comparisons, and summary tables).
- Recommended next steps: if convergence is slow, tighten ranges for the most sensitive parameters (e.g., LR C, RF max_depth/n_estimators) and allow GridAgent to exploit locally for 1–2 loops, then return to BayesianAgent for global refinement.
