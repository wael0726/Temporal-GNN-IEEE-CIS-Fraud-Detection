from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline import train_xgb
from src.data import load_ieee_cis
from src.evaluate import best_threshold, metrics
from src.features import build_temporal_features
from src.graph import build_temporal_graph
from src.graph_model import train_graphsage, train_mlp


RELATIONS = ["card1", "addr1", "DeviceInfo", "P_emaildomain", "ProductCD"]


def make_folds(n: int, n_folds: int = 4, train_fraction: float = .50,
               val_fraction: float = .15, test_fraction: float = .15, gap: int = 0):
    """Create expanding-window folds with a chronological gap before validation/test."""
    if train_fraction + val_fraction + test_fraction >= 1.0:
        raise ValueError("Fold fractions must leave room for later expanding windows.")
    folds = []
    span = int(n * test_fraction)
    initial_train = max(100, int(n * train_fraction))
    step = max(1, int(n * (1.0 - train_fraction - val_fraction - test_fraction) / max(n_folds - 1, 1)))
    for i in range(n_folds):
        train_end = initial_train + i * step
        val_start = train_end + gap
        val_end = val_start + max(50, int(n * val_fraction))
        test_start = val_end + gap
        test_end = test_start + span
        if test_end > n:
            break
        folds.append((0, train_end, val_start, val_end, test_start, test_end))
    return folds


def mask(n, start, end):
    out = torch.zeros(n, dtype=torch.bool)
    out[start:end] = True
    return out


def main():
    p = argparse.ArgumentParser(description="Expanding-window temporal robustness study.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-rows", type=int, default=50000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--gap", type=int, default=0)
    p.add_argument("--output-dir", default="outputs/walk_forward")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_ieee_cis(args.data_dir, max_rows=args.max_rows)
    df, features = build_temporal_features(df)
    n = len(df)
    folds = make_folds(n, args.n_folds, gap=args.gap)
    y = df.label.to_numpy(dtype=np.int8)
    rows = []

    for fold_id, (tr_start, tr_end, val_start, val_end, test_start, test_end) in enumerate(folds, 1):
        if y[tr_start:tr_end].sum() == 0 or y[val_start:val_end].sum() == 0 or y[test_start:test_end].sum() == 0:
            continue

        scaler = StandardScaler().fit(df.loc[tr_start:tr_end - 1, features])
        work = df.copy()
        work.loc[:, features] = scaler.transform(work[features]).astype("float32")
        X = work[features].to_numpy(dtype=np.float32)

        # XGBoost
        t0 = time.perf_counter()
        xgb = train_xgb(X[tr_start:tr_end], y[tr_start:tr_end], seed=42 + fold_id)
        xgb_seconds = time.perf_counter() - t0
        pv = xgb.predict_proba(X[val_start:val_end])[:, 1]
        pt = xgb.predict_proba(X[test_start:test_end])[:, 1]
        xgb_val = metrics(y[val_start:val_end], pv)["pr_auc"]
        xgb_test = metrics(
            y[test_start:test_end], pt,
            threshold=best_threshold(y[val_start:val_end], pv)
        )
        rows.append({
            "fold": fold_id, "model": "xgboost",
            "train": [tr_start, tr_end], "validation": [val_start, val_end],
            "test_window": [test_start, test_end],
            "gap": args.gap, "validation_pr_auc": xgb_val,
            "test": xgb_test, "train_seconds": xgb_seconds,
        })

        # Feature-only neural control.
        graph = build_temporal_graph(work.iloc[:test_end], features, RELATIONS, history_k=1)
        tr_mask = mask(test_end, tr_start, tr_end)
        va_mask = mask(test_end, val_start, val_end)
        mlp, history = train_mlp(
            graph, tr_mask, va_mask, epochs=args.epochs, hidden=96, lr=7e-4,
            dropout=.20, seed=42 + fold_id, patience=7
        )
        mlp.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(mlp(graph.x)).numpy()
        pv = p_all[val_start:val_end]
        pt = p_all[test_start:test_end]
        rows.append({
            "fold": fold_id, "model": "feature_mlp",
            "train": [tr_start, tr_end], "validation": [val_start, val_end],
            "test_window": [test_start, test_end],
            "gap": args.gap,
            "validation_pr_auc": metrics(y[val_start:val_end], pv)["pr_auc"],
            "test": metrics(
                y[test_start:test_end], pt,
                threshold=best_threshold(y[val_start:val_end], pv)
            ),
            "training_epochs": len(history),
        })

        # Causal graph model. Graph is built only through the test horizon; no
        # post-test events can become message-passing neighbors.
        gnn, history = train_graphsage(
            graph, tr_mask, va_mask, epochs=args.epochs, hidden=128, lr=7e-4,
            dropout=.15, seed=42 + fold_id, patience=7
        )
        gnn.eval()
        with torch.no_grad():
            p_all = torch.sigmoid(gnn(graph.x, graph.edge_index)).numpy()
        pv = p_all[val_start:val_end]
        pt = p_all[test_start:test_end]
        rows.append({
            "fold": fold_id, "model": "graphsage",
            "train": [tr_start, tr_end], "validation": [val_start, val_end],
            "test_window": [test_start, test_end],
            "gap": args.gap,
            "validation_pr_auc": metrics(y[val_start:val_end], pv)["pr_auc"],
            "test": metrics(
                y[test_start:test_end], pt,
                threshold=best_threshold(y[val_start:val_end], pv)
            ),
            "training_epochs": len(history),
            "edges": int(graph.edge_index.size(1)),
        })

    flat = []
    for r in rows:
        flat.append({
            "fold": r["fold"], "model": r["model"],
            "validation_pr_auc": r["validation_pr_auc"],
            "test_pr_auc": r["test"]["pr_auc"],
            "test_roc_auc": r["test"].get("roc_auc"),
            "test_ap_lift_vs_random": r["test"]["ap_lift_vs_random"],
            "test_precision_at_1pct": r["test"]["precision_at_1pct"],
            "test_recall_at_1pct": r["test"]["recall_at_1pct"],
            "brier": r["test"].get("brier"),
            "gap": r["gap"],
            "training_epochs": r.get("training_epochs"),
            "edges": r.get("edges"),
        })
    (out / "results.json").write_text(json.dumps({
        "protocol": {
            "folds": folds,
            "gap_rows": args.gap,
            "expanding_window": True,
            "test_never_used_for_selection": True,
        },
        "rows": rows,
    }, indent=2), encoding="utf-8")
    pd.DataFrame(flat).to_csv(out / "results.csv", index=False)

    if flat:
        summary = pd.DataFrame(flat).groupby("model").agg(
            mean_test_pr_auc=("test_pr_auc", "mean"),
            std_test_pr_auc=("test_pr_auc", "std"),
            mean_ap_lift=("test_ap_lift_vs_random", "mean"),
        ).reset_index()
        summary.to_csv(out / "summary.csv", index=False)

    print(json.dumps({
        "folds_completed": len(set(r["fold"] for r in rows)),
        "rows": len(rows),
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
