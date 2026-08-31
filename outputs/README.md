# Experiment outputs

This directory contains generated numerical results, model artifacts, and plots from different experiment protocols.

## Important rule: outputs are not interchangeable

Each subdirectory represents a distinct run or study. Do not replace an older result with a newer number merely because the newer run has a different configuration.

### `benchmark_50k/`

Bundled 50k benchmark artifacts, including model metrics, calibration artifacts, predictions, and plots.

### `ml_study_full/`

50k multi-model experimental study with:

- 3 seeds (`42, 123, 456`);
- main model controls;
- feature-group experiments;
- history-depth experiments;
- relation ablations;
- paired GraphSAGE vs XGBoost comparisons.

The CSV contains 38 rows because the study includes more than the five canonical main-model runs.

### `walk_forward/`

Four-fold expanding-window temporal evaluation on 50k rows, executed with 10 neural-model epochs and gap 0.

Important files:

```text
results.csv
summary.csv
results.json
walk_forward_pr_auc.png
```

### Other directories

The repository also contains earlier smoke tests, 10k studies, and reproduction runs. They are retained as experiment history and should not be presented as the final benchmark unless the corresponding protocol is explicitly stated.

## Reproducibility

Main commands are documented in:

- `README.md`
- `EXPERIMENTS.md`
- `RESEARCH_PROTOCOL.md`

The project intentionally distinguishes **implemented experiment machinery** from **numerical evidence actually executed and stored**.
