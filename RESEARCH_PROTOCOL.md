# Experimental Protocol — 2026

This repository is evaluated as an ML research project, not only as a software demo.

## Research question

> **Does historical relational structure between transactions add predictive information beyond causal temporal/tabular features for fraud detection?**

A secondary engineering question is how much compute and graph size are required for any observed relational benefit.

---

## Primary endpoint

**Average Precision (AP / PR-AUC)** on chronologically held-out transactions.

Fraud is highly imbalanced, so AP is always reported with test prevalence and AP lift over a random ranking baseline.

### Secondary endpoints

- ROC-AUC;
- Precision@0.1%, 0.5%, 1%, 2%, 5%;
- Recall at the same alert budgets;
- F1 at a validation/calibration-selected threshold;
- Brier score;
- log loss;
- ECE using 10 bins.

---

# Data and leakage policy

1. Transactions are sorted by `TransactionDT` and `TransactionID` before truncation.
2. Historical features use only prior events.
3. Missing entity values never form a shared graph entity.
4. Graph edges point strictly from a past transaction to a current transaction.
5. Duplicate transaction-pair edges are removed.
6. The scaler is fitted on the training segment only.
7. The final test window is never used to select model hyperparameters.
8. Operating thresholds are selected before final test reporting.
9. Optional strict calibration splits the validation period chronologically so that model selection and calibration do not reuse the same observations.
10. Walk-forward evaluation builds the graph only through the current test horizon; no post-test transaction is allowed to become a message-passing neighbor.

The temporal graph therefore represents information that could have existed at prediction time.

---

# Model ladder

The minimum comparison is:

1. **Logistic Regression** — linear control.
2. **HistGradientBoosting** — nonlinear tabular control.
3. **XGBoost** — strong production-style tabular baseline.
4. **Feature MLP** — same causal features without graph message passing.
5. **GraphSAGE** — causal relational model.

This ladder separates linear, nonlinear-tabular, neural-feature, and relational signal.

---

# Graph construction

Each transaction is a node.

Current relation columns:

```text
card1
addr1
DeviceInfo
P_emaildomain
ProductCD
```

With `history_k=k`, a transaction connects to up to its `k` most recent known historical transactions per relation.

Edges are directed:

```text
historical transaction  ─────►  current transaction
```

A future transaction cannot send a message to an earlier transaction.

Missing values are not treated as real entities. This prevents all missing `DeviceInfo`, for example, from becoming one giant artificial component.

---

# Feature construction

The causal feature set includes:

- `log_amount`;
- cyclic time features;
- previous entity occurrence counts;
- previous one-hour entity counts;
- seconds since previous `card1` activity;
- previous 24-hour `card1` activity.

Current-event information is excluded from historical counts.

---

# Robustness experiments

## Seeds

Default research seeds:

```text
42, 123, 456, 789, 2026
```

Report per-seed AP and mean ± standard deviation.

The currently bundled full study was run with the first three seeds only:

```text
42, 123, 456
```

Therefore it is a **3-seed result**, not yet the planned 5-seed robustness campaign.

## Graph history

Test:

```text
k = 1, 3, 5, 10
```

## Relations

Test:

- all relations;
- each relation alone;
- leave-one-relation-out.

Relations:

```text
card1, addr1, DeviceInfo, P_emaildomain, ProductCD
```

## Feature groups

Compare:

- amount + cyclic time features;
- behavioral historical features;
- all causal features.

The purpose is to determine whether GraphSAGE adds value beyond the same temporal/statistical information available to tabular models.

---

# Temporal robustness

The preferred temporal validation is an expanding-window walk-forward protocol.

The currently executed run used:

```text
max rows = 50,000
epochs = 10
folds = 4
gap = 0
```

The four folds are:

| Fold | Train | Validation | Test |
|---:|---|---|---|
| 1 | 0–24,999 | 25,000–32,499 | 32,500–39,999 |
| 2 | 0–28,332 | 28,333–35,832 | 35,833–43,332 |
| 3 | 0–31,665 | 31,666–39,165 | 39,166–46,665 |
| 4 | 0–34,998 | 34,999–42,498 | 42,499–49,998 |

A chronological gap can be enabled for future experiments.

### Important interpretation rule

The current 10-epoch walk-forward result is not a claim about the maximum potential of GraphSAGE. It is a result for the exact configuration that was executed.

A controlled 30-epoch follow-up should preserve the same folds and data protocol:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 30 --output-dir outputs\walk_forward_30
```

---

# Calibration

Calibration must use validation/calibration data only.

Compare:

- raw probabilities;
- sigmoid/Platt calibration;
- isotonic calibration when the calibration window is sufficiently large.

Calibration is not a ranking experiment. A monotonic calibration method should not materially change AP because it preserves score ordering.

Strict calibration should split validation chronologically:

```text
early validation → model selection
later validation → calibration + threshold selection
final test       → untouched evaluation
```

---

# Statistical comparison

When two models are evaluated on the same test events, compare them with a **paired bootstrap of the AP difference** rather than treating their confidence intervals as independent.

Report:

- ΔAP;
- 95% bootstrap interval;
- bootstrap probability that model B outperforms model A.

A result whose interval crosses zero is treated as inconclusive rather than as proof of superiority.

---

# Scalability

The intended scalability range is:

```text
10k / 50k / 100k rows
```

Measure:

- elapsed time;
- process RSS / memory change;
- graph edge count;
- effective graph mode;
- AP and AP lift;
- XGBoost vs GraphSAGE performance.

`graph-mode=auto` can select full-batch or PyG neighbor-sampled execution according to the configured threshold.

The repository currently contains the scalability machinery but does not claim a completed 100k numerical result in this release.

---

# Interpretation standard

A single high score is not sufficient evidence.

The final conclusion should distinguish:

- average performance;
- variance across seeds;
- variance across temporal windows;
- contribution of graph structure;
- calibration quality;
- operational alert-budget performance;
- compute and memory cost.

The project must not be tuned simply to make GraphSAGE beat XGBoost. If XGBoost remains better after fair training and ablation, that is a valid and useful result.

---

# Recommended execution order

## 1. Automated tests

```powershell
pytest -q
```

## 2. Fast ML sanity check

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 10000 --epochs 5 --suite core --seeds 42 --output-dir outputs\ml_study_core
```

## 3. Controlled ablations

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 15 --suite ablation --seeds 42 --output-dir outputs\ml_study_ablation
```

## 4. Multi-seed study

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 20 --suite full --seeds 42 123 456 --output-dir outputs\ml_study_full
```

## 5. Walk-forward robustness

Current executed run:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 10 --output-dir outputs\walk_forward
```

Recommended controlled follow-up:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 30 --output-dir outputs\walk_forward_30
```

## 6. Full 5-seed robustness campaign

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 20 --suite robustness --seeds 42 123 456 789 2026 --output-dir outputs\ml_study_robustness_50k
```

## 7. 100k scalability

```powershell
python scripts/run_scalability.py --data-dir data --sizes 10000 50000 100000 --epochs 20 --graph-mode auto
```

## 8. Strict calibration comparison

```powershell
python scripts/run_experiment.py --data-dir data --dataset ieee_cis --max-rows 100000 --epochs 40 --history-k 3 --tune --graph-mode auto --strict-calibration --compare-isotonic --output-dir outputs\benchmark_100k_strict
```

The final 100k result should only be interpreted after the robustness and ablation studies are established.
