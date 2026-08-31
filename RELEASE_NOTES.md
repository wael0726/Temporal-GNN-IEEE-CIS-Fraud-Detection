# Release Notes — Hardened Research Build

## 2026 research hardening

This build extends the previous fraud-detection repository into a more explicit experimental ML framework. The documentation now distinguishes implemented methodology from numerical evidence that has actually been executed.

### Added / retained

- Logistic Regression and HistGradientBoosting controls.
- Feature-group ablations.
- Relation-only and leave-one-relation-out GraphSAGE ablations.
- Multi-seed robustness summaries.
- Expanding-window temporal evaluation with optional gaps.
- 10k / 50k / 100k scalability runner.
- Process-memory and wall-clock measurements.
- Strict chronological calibration mode.
- Isotonic calibration comparison with minimum-data guard.
- Log loss and paired-bootstrap `P(B > A)` reporting.
- Retained-vs-raw graph relation edge accounting.
- Per-edge historical time deltas for graph auditing/future time-aware models.
- Formal research protocol in `RESEARCH_PROTOCOL.md`.
- Walk-forward PR-AUC visualization in `outputs/walk_forward/walk_forward_pr_auc.png`.
- Updated GitHub-facing documentation in `README.md`, `EXPERIMENTS.md`, and `outputs/README.md`.

## Executed evidence included in this release

### 50k full ML study

Command:

```powershell
python scripts/run_ml_study.py --data-dir data --max-rows 50000 --epochs 20 --suite full --seeds 42 123 456 --output-dir outputs\ml_study_full
```

The output contains 38 rows across canonical models and controlled experiments. The canonical main-model comparison contains three seeds per model.

### Walk-forward temporal evaluation

Command:

```powershell
python scripts/run_temporal_cv.py --data-dir data --max-rows 50000 --epochs 10 --output-dir outputs\walk_forward
```

Result:

- 4 folds completed;
- expanding chronological windows;
- gap = 0;
- 10 epochs for GraphSAGE and Feature MLP.

The current four-fold mean test AP is:

| Model | Mean test AP |
|---|---:|
| XGBoost | **0.06427** |
| Feature MLP | 0.04733 |
| GraphSAGE | 0.04456 |

### Test suite

The hardened repository was previously validated with:

```text
14 passed
```

## Important non-claims

The following are **not** claimed as completed final evidence in this release:

- a five-seed 50k robustness campaign;
- a 30-epoch walk-forward run;
- a completed 100k numerical benchmark;
- a final relation/history configuration selected across all temporal folds.

The repository contains the machinery and commands for these experiments, but documentation does not invent their results.

## Interpretation

The current evidence is intentionally mixed. An earlier single-split benchmark favored GraphSAGE, while the current walk-forward evaluation favors XGBoost. The 50k three-seed full study gives GraphSAGE a higher mean AP than XGBoost on its canonical rows, but with greater variability.

The project therefore makes no universal claim that GraphSAGE is superior. The next controlled experiment is to repeat the walk-forward protocol with a longer GraphSAGE training budget before making any architectural conclusion.
