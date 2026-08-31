# Experiment artifacts

This directory contains documentation figures that are safe to reference from GitHub. Numerical experiment outputs live under `outputs/`.

## Static documentation figures

- `static/pipeline_overview.png` — high-level pipeline diagram.
- `static/temporal_graph_concept.png` — conceptual causal temporal graph.

## Generated experiment figures

Experiment-specific plots are stored alongside their numerical outputs, for example:

```text
outputs/benchmark_50k/model_comparison.png
outputs/benchmark_50k/precision_recall.png
outputs/benchmark_50k/fraud_rate_over_time.png
outputs/walk_forward/walk_forward_pr_auc.png
```

These images can be embedded directly in `README.md` with relative Markdown paths.

Model files and raw numerical outputs are intentionally kept under `outputs/` rather than duplicated here, so a reader can distinguish documentation assets from experiment artifacts.
