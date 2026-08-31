# ML Experiment Report

This document records the experimental evidence currently bundled with the repository. Results are separated by protocol so that a later experiment does not silently overwrite an earlier benchmark.

---

## 1. Research question

> **Does historical relational structure between transactions add predictive information beyond causal temporal/tabular features for fraud detection?**

The project is therefore evaluated as a model-comparison study rather than as a demonstration that GraphSAGE must win.

## 2. Evaluation ladder

| Model | Purpose |
|---|---|
| Logistic Regression | Linear control |
| HistGradientBoosting | Nonlinear tabular control |
| XGBoost | Strong production-style tabular baseline |
| Feature MLP | Neural control using the same causal features without graph message passing |
| GraphSAGE | Causal relational model using the historical transaction graph |

The feature-only MLP is particularly important because it separates generic neural-network effects from the contribution of graph message passing.

---

# 3. Historical 50k reference benchmark

An earlier reference table in the repository recorded:

| Model | Validation AP | Test AP | Test ROC-AUC | P@1% | R@1% |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.1306 | 0.0371 | 0.5987 | 0.080 | 0.04196 |
| GraphSAGE | 0.0963 | 0.0642 | 0.6787 | 0.160 | 0.08392 |

Test prevalence in that reference table was 0.01907.

The repository also contains a `outputs/benchmark_50k/metrics.json` artifact whose bundled run reports GraphSAGE test AP 0.06034 and XGBoost test AP 0.03705. These should not be silently reconciled: they are separate stored benchmark artifacts from different runs/configurations.

The important rule is that **the historical reference is retained as historical evidence** and is not overwritten by the newer multi-seed or walk-forward studies.

---

# 4. 50k full ML study

Command used:

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 20 --suite full --seeds 42 123 456 --output-dir outputs\ml_study_full
```

The output contains **38 rows**, including canonical model runs, feature-group controls, history-depth experiments, relation ablations, and paired comparisons.

## 4.1 Canonical main-model results

For the table below, only one canonical row per seed is used. GraphSAGE is the all-relation configuration with `history_k=3`.

| Model | Seeds | Mean test AP | Std. dev. | Mean ROC-AUC | Mean P@1% | Mean R@1% |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 3 | **0.05525** | 0.00000 | 0.6607 | 0.1200 | 0.0629 |
| GraphSAGE | 3 | **0.05194** | 0.00900 | 0.6491 | 0.1067 | 0.0560 |
| HistGradientBoosting | 3 | 0.03758 | 0.00107 | 0.6645 | 0.0533 | 0.0280 |
| XGBoost | 3 | 0.03532 | 0.00230 | 0.5879 | 0.0756 | 0.0396 |
| Feature MLP | 3 | 0.02738 | 0.00386 | 0.5857 | 0.0400 | 0.0210 |

### Per-seed AP

| Model | Seed 42 | Seed 123 | Seed 456 |
|---|---:|---:|---:|
| Logistic Regression | 0.05525 | 0.05525 | 0.05525 |
| HistGradientBoosting | 0.03725 | 0.03672 | 0.03878 |
| XGBoost | 0.03705 | 0.03620 | 0.03270 |
| Feature MLP | 0.02573 | 0.02463 | 0.03179 |
| GraphSAGE | 0.04156 | 0.05724 | 0.05704 |

### Paired GraphSAGE − XGBoost AP comparison

The study reports a paired bootstrap comparison on the same test events:

| Seed | ΔAP | 95% bootstrap interval | P(GraphSAGE > XGBoost) |
|---:|---:|---:|---:|
| 42 | +0.00451 | [-0.01382, 0.02114] | 0.671 |
| 123 | +0.02104 | [0.00034, 0.04950] | 0.978 |
| 456 | +0.02434 | [0.00561, 0.05121] | 1.000 |

These results are supportive of GraphSAGE on the paired 50k study for seeds 123 and 456, while seed 42 is inconclusive because its bootstrap interval crosses zero.

---

# 5. Feature-group controls

The full study includes three feature regimes:

- `base_temporal_amount`
- `behavioral_history`
- all causal features

The purpose is to determine whether apparent graph performance is simply coming from the historical/statistical features that are also supplied to the tabular models.

For seed 42, the canonical all-feature configuration produced:

| Model | Test AP |
|---|---:|
| XGBoost | 0.03705 |
| Feature MLP | 0.02573 |
| GraphSAGE | 0.04156 |

For the base temporal/amount group:

| Model | Test AP |
|---|---:|
| XGBoost | 0.02684 |
| Feature MLP | 0.03152 |
| GraphSAGE | 0.03294 |

For the behavioral-history group:

| Model | Test AP |
|---|---:|
| XGBoost | 0.02674 |
| Feature MLP | 0.02965 |
| GraphSAGE | 0.04150 |

These are controlled feature-group observations, not evidence that one feature group is universally optimal.

---

# 6. Graph history-depth ablation

For seed 42 and all five relations:

| History `k` | Validation AP | Test AP | Edges |
|---:|---:|---:|---:|
| 1 | 0.06828 | 0.03357 | 176,433 |
| 3 | 0.08575 | 0.04156 | 519,889 |
| 5 | 0.09309 | 0.04142 | 858,405 |
| 10 | **0.09806** | **0.04206** | 1,687,356 |

The main observation is that larger history increases graph size substantially. The test AP gain from `k=3` to `k=10` is comparatively small, so the additional graph cost must be justified by later robustness experiments before changing the default.

---

# 7. Relation ablations

The runner includes relation-only and leave-one-relation-out experiments for:

```text
card1
addr1
DeviceInfo
P_emaildomain
ProductCD
```

For seed 42, selected leave-one-relation-out results show that the graph signal is distributed across relations rather than being attributable to one single edge type. Exact values are preserved in:

```text
outputs/ml_study_full/results.csv
outputs/ml_study_full/results.json
```

The ablation results should be interpreted using validation AP for selection, with test AP retained for final reporting rather than repeatedly selecting the best relation from the test set.

---

# 8. Walk-forward temporal robustness

Command used:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 10 --output-dir outputs\walk_forward
```

Protocol:

- 4 expanding chronological folds;
- 50k maximum rows;
- gap = 0;
- test windows are later in time than the corresponding training windows;
- GraphSAGE and Feature MLP trained for 10 epochs;
- XGBoost uses its standard runner configuration;
- test windows are not used for model selection.

## 8.1 Fold protocol

| Fold | Train | Validation | Test |
|---:|---|---|---|
| 1 | rows 0–24,999 | 25,000–32,499 | 32,500–39,999 |
| 2 | rows 0–28,332 | 28,333–35,832 | 35,833–43,332 |
| 3 | rows 0–31,665 | 31,666–39,165 | 39,166–46,665 |
| 4 | rows 0–34,998 | 34,999–42,498 | 42,499–49,998 |

## 8.2 Summary

| Model | Mean test AP | Std. dev. | Mean AP lift |
|---|---:|---:|---:|
| XGBoost | **0.06427** | 0.02809 | **2.55×** |
| Feature MLP | 0.04733 | 0.01725 | 1.98× |
| GraphSAGE | 0.04456 | 0.01543 | 1.79× |

## 8.3 Fold-level AP

| Fold | XGBoost | Feature MLP | GraphSAGE |
|---:|---:|---:|---:|
| 1 | **0.07927** | 0.06791 | 0.06432 |
| 2 | **0.09523** | 0.03976 | 0.04584 |
| 3 | **0.04934** | 0.02802 | 0.04112 |
| 4 | 0.03327 | **0.05363** | 0.02695 |

The current temporal run therefore favors XGBoost. GraphSAGE remains above the random baseline in every fold, but does not beat XGBoost in any fold.

### Important limitation

This walk-forward run uses **10 epochs** for GraphSAGE. The result is therefore a robustness result for the current 10-epoch configuration, not a final statement about the architecture's maximum attainable performance.

The next controlled experiment is a 30-epoch walk-forward run under the same protocol:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 30 --output-dir outputs\walk_forward_30
```

No result from that future run is claimed here.

---

# 9. Operational alert budgets

The temporal study reports fixed alert-budget metrics at 0.1%, 0.5%, 1%, 2%, and 5%.

At the 1% budget:

| Fold | Model | Precision | Recall | AP lift |
|---:|---|---:|---:|---:|
| 1 | XGBoost | 22.67% | 8.29% | 2.90× |
| 1 | GraphSAGE | 12.00% | 4.39% | 2.35× |
| 2 | XGBoost | 25.33% | 9.31% | 3.50× |
| 2 | GraphSAGE | 6.67% | 2.45% | 1.69× |
| 3 | XGBoost | 12.00% | 4.97% | 2.04× |
| 3 | GraphSAGE | 5.33% | 2.21% | 1.70× |
| 4 | XGBoost | 9.33% | 4.90% | 1.74× |
| 4 | GraphSAGE | 1.33% | 0.70% | 1.41× |

---

# 10. Calibration

Calibration is evaluated separately from ranking. Validation-only Platt calibration does not change the ordering of predictions, so AP should remain unchanged while probability quality can improve.

The bundled `outputs/benchmark_50k/metrics.json` reports:

| Model | Raw Brier | Calibrated Brier | Raw ECE | Calibrated ECE |
|---|---:|---:|---:|---:|
| XGBoost | 0.02651 | 0.01874 | 0.0432 | 0.0089 |
| GraphSAGE | 0.13402 | 0.01854 | 0.3111 | 0.0062 |

This is a useful example of why ranking and calibration should not be conflated.

---

# 11. Graph size and compute

The 50k bundled benchmark used:

- 50,000 transaction nodes;
- 519,889 raw/deduplicated graph edges reported by the benchmark;
- 5 relation types;
- `history_k=3`;
- causal past-to-current edge direction.

The walk-forward graph edge counts grew with the expanding training horizon:

| Fold | GraphSAGE edges |
|---:|---:|
| 1 | 139,582 |
| 2 | 151,811 |
| 3 | 164,147 |
| 4 | 176,428 |

Increasing `history_k` also increases graph size rapidly, which is one reason scalability and accuracy must be evaluated together.

---

# 12. Current scientific conclusion

The evidence does not support a universal claim that GraphSAGE is better than XGBoost.

Instead:

1. the historical single-split benchmark favored GraphSAGE;
2. the 50k three-seed study still gives GraphSAGE a higher mean AP than XGBoost on the canonical rows, with substantial seed variability;
3. the 4-fold walk-forward study currently favors XGBoost;
4. GraphSAGE remains above the random baseline on all four temporal test windows;
5. feature-only MLP results show that a generic neural network is not consistently sufficient to explain the observed performance;
6. history depth changes GraphSAGE behavior and graph cost, but the best setting is not yet established across multiple temporal folds.

The correct next question is therefore not "how do we make GraphSAGE win?" but:

> **Under the same temporal protocol, does a better-trained and better-structured relational model provide a stable advantage over strong tabular baselines?**

---

# 13. Remaining experiments

The repository contains the machinery for:

- 5-seed robustness (`42, 123, 456, 789, 2026`);
- history-depth experiments (`k=1,3,5,10`);
- relation ablations;
- feature-group controls;
- strict chronological calibration;
- isotonic calibration when enough calibration data exist;
- paired bootstrap model comparisons;
- 10k / 50k / 100k scalability runs.

Not all of these are complete final evidence yet. The README and this report deliberately distinguish implemented machinery from executed numerical results.
