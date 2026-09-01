# Temporal Graph Learning for IEEE-CIS Fraud Detection

> A research-style fraud detection project comparing a causal temporal GraphSAGE model with strong tabular baselines on the IEEE-CIS dataset.

The project asks a specific question:

> **Does historical relational structure between transactions add predictive information beyond causal temporal/tabular features for fraud detection?**

The answer is evaluated empirically rather than assuming that the graph model must outperform the tabular baseline.

---

## Project overview

Each transaction is represented as a node. Historical relationships through `card1`, `addr1`, `DeviceInfo`, `P_emaildomain`, and `ProductCD` create directed **past → current** graph edges. The model only receives information that would have been available before the transaction being scored.

The experimental ladder contains:

1. Logistic Regression — linear control
2. HistGradientBoosting — nonlinear tabular control
3. XGBoost — strong production-style tabular baseline
4. Feature-only MLP — same causal features without message passing
5. GraphSAGE — causal relational model

This separation is important: if GraphSAGE improves over the feature-only MLP, the gain is more plausibly attributable to relational message passing rather than simply using a neural network.

![Temporal graph concept](artifacts/static/temporal_graph.png)

---

## Methodological safeguards

The implementation is designed around temporal leakage prevention:

- transactions are sorted by `TransactionDT` and `TransactionID` before truncation;
- historical features use only prior events;
- missing categorical values never become shared graph entities;
- graph edges point strictly from past transactions to current transactions;
- duplicate edges are removed;
- the feature scaler is fitted on the training segment only;
- model and threshold selection are validation-only;
- the final test window is not used for model selection;
- walk-forward evaluation keeps later test events from becoming neighbors of earlier test events.

The temporal aspect is therefore **causal historical feature engineering + a directed temporal graph**. This is not presented as a full Temporal Graph Network (TGN) with persistent event memory.

![Pipeline overview](artifacts/static/pipelineoverview.png)

---

# Results

Fraud is highly imbalanced, so the headline ranking metric is **Average Precision (AP / PR-AUC)** rather than accuracy. AP is reported together with fraud prevalence and lift over a random-ranking baseline.

## 50k multi-model study — 3 seeds

The complete `ml_study_full` output contains 38 experiment rows because it includes the main models plus feature-group, history-depth, relation, and paired-comparison experiments. The table below shows only the **canonical main-model rows**: one result per seed for each model, with GraphSAGE using all five relations and `history_k=3`.

| Model | Seeds | Mean test AP | Std. dev. | Mean ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 3 | **0.05525** | 0.00000 | 0.6607 |
| GraphSAGE | 3 | **0.05194** | 0.00900 | 0.6491 |
| HistGradientBoosting | 3 | 0.03758 | 0.00107 | 0.6645 |
| XGBoost | 3 | 0.03532 | 0.00230 | 0.5879 |
| Feature-only MLP | 3 | 0.02738 | 0.00386 | 0.5857 |

**Interpretation:** GraphSAGE is competitive in the 50k multi-seed study and has a higher mean AP than XGBoost on these canonical rows, but Logistic Regression has the highest mean AP. GraphSAGE also shows noticeably more seed-to-seed variability than XGBoost.

This study should **not** be treated as a direct replacement for the temporal walk-forward benchmark: the protocols and selection procedures differ.

---

## Walk-forward temporal evaluation — 4 folds

The latest walk-forward run uses 50k chronologically ordered rows, 4 expanding folds, no gap, and a maximum neural training budget of **30 epochs**. Early stopping/runner behavior can result in fewer actual epochs on individual folds.

| Model | Mean test AP | Std. dev. | Mean AP lift vs random |
|---|---:|---:|---:|
| XGBoost | **0.06427** | 0.02809 | **2.55×** |
| GraphSAGE | 0.06148 | 0.02444 | 2.44× |
| Feature-only MLP | 0.06079 | 0.01834 | 2.49× |

The fold-level results are:

| Fold | XGBoost AP | GraphSAGE AP | Feature MLP AP |
|---:|---:|---:|---:|
| 1 | 0.07927 | 0.08426 | 0.08513 |
| 2 | 0.09523 | 0.06502 | 0.06264 |
| 3 | 0.04934 | 0.06969 | 0.04177 |
| 4 | 0.03327 | 0.02695 | 0.05363 |

![Walk-forward PR-AUC](outputs/walk_forward/walk_forward_prauc.png)

**Interpretation:** XGBoost remains the strongest model on mean test AP across the four temporal folds, but the gap is now small: GraphSAGE reaches 0.06148 mean AP versus 0.06427 for XGBoost. GraphSAGE also beats XGBoost on fold 1 and fold 3, while the feature-only MLP wins fold 1 and fold 4. This is substantially more competitive than the earlier 10-epoch walk-forward run.

GraphSAGE remains above the random baseline on every fold. The results therefore do **not** support a universal GraphSAGE advantage, but they also do not justify dismissing the relational model: performance is competitive and varies substantially across time windows.

### Training budget matters

The earlier 10-epoch run produced lower GraphSAGE mean AP (0.04456). The new maximum-30-epoch run reaches 0.06148. This is a large improvement under the same four-fold temporal protocol and is evidence that the earlier result was partly limited by the training budget.

The latest run also shows that GraphSAGE does not necessarily require all 30 epochs on every fold: the recorded training epochs were 30, 25, 30, and 11 respectively.

---

## Operational fraud-review metrics

The project also reports precision and recall under fixed alert budgets. This is more meaningful than accuracy for a highly imbalanced fraud problem because a real review system can only investigate a limited fraction of transactions.

For the latest walk-forward evaluation, the 1% alert-budget results were:

| Fold | Model | Precision@1% | Recall@1% | AP lift |
|---:|---|---:|---:|---:|
| 1 | XGBoost | 22.67% | 8.29% | 2.90× |
| 1 | GraphSAGE | 22.67% | 8.29% | 3.08× |
| 2 | XGBoost | 25.33% | 9.31% | 3.50× |
| 2 | GraphSAGE | 13.33% | 4.90% | 2.39× |
| 3 | XGBoost | 12.00% | 4.97% | 2.04× |
| 3 | GraphSAGE | 16.00% | 6.63% | 2.89× |
| 4 | XGBoost | 9.33% | 4.90% | 1.74× |
| 4 | GraphSAGE | 1.33% | 0.70% | 1.41× |

The full CSV/JSON artifacts retain the 0.1%, 0.5%, 1%, 2%, and 5% alert-budget metrics.

---

## 50k benchmark visualization

The repository also contains the earlier 50k benchmark artifacts:

![Model comparison](outputs/benchmark_50k/model_comparison.png)

![Precision-recall comparison](outputs/benchmark_50k/precision_recall.png)

![Fraud rate over time](outputs/benchmark_50k/fraud_rate_over_time.png)

These figures are useful for visual interpretation, but the historical benchmark, multi-seed study, and walk-forward study are kept conceptually separate because they were produced under different experimental protocols.

---

## GraphSAGE history-depth study

The 50k full study also tested the number of historical neighbors retained per relation. For seed 42, using all five relations:

| History `k` | Validation AP | Test AP |
|---:|---:|---:|
| 1 | 0.06828 | 0.03357 |
| 3 | 0.08575 | 0.04156 |
| 5 | 0.09309 | 0.04142 |
| 10 | **0.09806** | **0.04206** |

The result suggests that increasing historical context can improve validation performance, but the test improvement from `k=3` to `k=10` is small. This is why history depth should be evaluated as a controlled ablation rather than selected from the test set.

---

## Calibration

Ranking quality and probability calibration are treated separately.

In the bundled 50k benchmark artifact, validation-only Platt calibration left AP unchanged, as expected, while substantially improving probability calibration:

| Model | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| XGBoost | 0.02651 | 0.01874 | 0.0432 | 0.0089 |
| GraphSAGE | 0.13402 | 0.01854 | 0.3111 | 0.0062 |

Calibration parameters are fitted without using the final test labels for fitting or selection.

---

## What the current evidence actually says

The project intentionally does not force a GraphSAGE victory.

The current evidence is mixed:

- the older single-split 50k benchmark showed GraphSAGE ahead of XGBoost on test AP;
- the 50k multi-seed study gives GraphSAGE a higher mean AP than XGBoost on its canonical rows, but Logistic Regression is highest and GraphSAGE has greater variability;
- the latest 4-fold walk-forward study favors XGBoost on mean AP, but only narrowly (0.06427 vs 0.06148);
- the new 30-epoch-budget walk-forward run substantially improves GraphSAGE over the earlier 10-epoch configuration (0.06148 vs 0.04456 mean AP);
- GraphSAGE remains above the random AP baseline in every walk-forward fold;
- the feature-only MLP is highly competitive in the latest temporal run, which means relational message passing is not the only source of useful signal;
- history-depth experiments show that graph context matters, but the best setting is not yet established across multiple temporal folds.

This is a more defensible conclusion than claiming that a GNN is automatically superior to a strong tabular baseline.

---

# Reproducibility

## Test suite

```powershell
pytest -q
```

The hardened repository's documented test suite was previously validated with **14 passing tests**.

## Multi-model study

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 20 --suite full --seeds 42 123 456 --output-dir outputs\ml_study_full
```

## Walk-forward temporal evaluation — latest configuration

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 30 --output-dir outputs\walk_forward
```

## 100k benchmark

A 100k benchmark is supported but **has not been numerically reported as final evidence in this repository**. Do not treat a smoke test as a final 100k result.

```powershell
New-Item -ItemType Directory -Force outputs\benchmark_100k
python scripts/run_experiment.py --data-dir data --dataset ieee_cis --max-rows 100000 --epochs 30 --history-k 3 --tune --graph-mode auto --output-dir outputs\benchmark_100k
```

For large graphs, `graph-mode=auto` can use PyTorch Geometric neighbor sampling when configured to do so.

---

# Repository documentation

- [`EXPERIMENTS.md`](EXPERIMENTS.md) — detailed experimental record and result interpretation
- [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) — predefined research protocol and anti-leakage rules
- [`RELEASE_NOTES.md`](RELEASE_NOTES.md) — changes introduced by the hardened research build
- [`outputs/README.md`](outputs/README.md) — explanation of generated experiment artifacts

---

# Project structure

```text
.
├── api/                    # API layer
├── artifacts/static/       # Documentation figures
├── data/                   # Dataset files / data instructions
├── outputs/                # Experiment outputs and plots
├── scripts/                # Reproducible experiment runners
├── src/                    # Feature, graph, model, training and evaluation code
├── tests/                  # Automated tests
├── EXPERIMENTS.md
├── RESEARCH_PROTOCOL.md
├── RELEASE_NOTES.md
└── README.md
```

---

## Scope and limitations

This is a research/portfolio implementation, not a production fraud service. The reported numbers depend on the IEEE-CIS data subset, temporal windows, feature construction, graph relations, hyperparameters, seeds, and compute environment.

The strongest remaining empirical questions are whether the improved GraphSAGE performance remains stable with additional seeds and temporal folds, which graph relations and history depths are genuinely useful, and whether graph message passing provides a statistically robust advantage over the strongest tabular and feature-only neural baselines.
