from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baseline import train_xgb
from src.data import load_ieee_cis
from src.evaluate import (
    best_threshold,
    bootstrap_ci,
    metrics,
    paired_bootstrap_delta,
)
from src.features import build_temporal_features
from src.graph import build_temporal_graph
from src.graph_model import train_graphsage, train_mlp, set_seed


ALL_RELATIONS = ["card1", "addr1", "DeviceInfo", "P_emaildomain", "ProductCD"]
BASE_FEATURE_PREFIXES = ("log_amount", "hour_sin", "hour_cos", "week_cycle_sin", "week_cycle_cos")


def chronological_split(n: int):
    if n < 100:
        raise ValueError("Need at least 100 rows.")
    return int(.70 * n), int(.85 * n)


def prepare(data_dir: str, max_rows: int):
    df = load_ieee_cis(data_dir, max_rows=max_rows)
    df, features = build_temporal_features(df)
    train_end, val_end = chronological_split(len(df))
    scaler = StandardScaler().fit(df.loc[:train_end - 1, features])
    df.loc[:, features] = scaler.transform(df[features]).astype("float32")
    y = df.label.to_numpy(dtype=np.int8)
    masks = {
        "train": np.arange(0, train_end, dtype=np.int64),
        "validation": np.arange(train_end, val_end, dtype=np.int64),
        "test": np.arange(val_end, len(df), dtype=np.int64),
    }
    for name, idx in masks.items():
        if y[idx].sum() == 0:
            raise ValueError(f"{name} split contains no positive examples.")
    return df, features, y, masks


def _mask_tensor(n, idx):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.as_tensor(idx, dtype=torch.long)] = True
    return mask


def feature_sets(features):
    base = [f for f in features if f in BASE_FEATURE_PREFIXES]
    behavior = [f for f in features if f not in base]
    return {
        "base_temporal_amount": base,
        "behavioral_history": behavior,
        "all_features": list(features),
    }


def _result_row(model_name, seed, validation, test, ci, **extra):
    row = {
        "model": model_name,
        "seed": int(seed),
        "validation_pr_auc": float(validation),
        "test": test,
        "ci": ci,
    }
    row.update(extra)
    return row


def run_xgb(df, features, y, masks, seed, params=None):
    X = df[features].to_numpy(dtype=np.float32)
    tr, va, te = masks["train"], masks["validation"], masks["test"]
    t0 = time.perf_counter()
    model = train_xgb(X[tr], y[tr], seed=seed, params=params)
    train_seconds = time.perf_counter() - t0
    pv = model.predict_proba(X[va])[:, 1]
    pt = model.predict_proba(X[te])[:, 1]
    threshold = best_threshold(y[va], pv)
    return _result_row(
        "xgboost", seed,
        metrics(y[va], pv)["pr_auc"],
        metrics(y[te], pt, threshold=threshold),
        bootstrap_ci(y[te], pt, "pr_auc", 500, seed),
        features=features,
        train_seconds=train_seconds,
        _test_y=y[te].tolist(),
        _test_proba=pt.tolist(),
    )


def run_hgb(df, features, y, masks, seed):
    X = df[features].to_numpy(dtype=np.float32)
    tr, va, te = masks["train"], masks["validation"], masks["test"]
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=350, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=seed,
    )
    model.fit(X[tr], y[tr])
    train_seconds = time.perf_counter() - t0
    pv = model.predict_proba(X[va])[:, 1]
    pt = model.predict_proba(X[te])[:, 1]
    return _result_row(
        "hist_gradient_boosting", seed,
        metrics(y[va], pv)["pr_auc"],
        metrics(y[te], pt, threshold=best_threshold(y[va], pv)),
        bootstrap_ci(y[te], pt, "pr_auc", 500, seed),
        features=features, train_seconds=train_seconds,
        _test_y=y[te].tolist(), _test_proba=pt.tolist(),
    )


def run_logistic(df, features, y, masks, seed):
    X = df[features].to_numpy(dtype=np.float32)
    tr, va, te = masks["train"], masks["validation"], masks["test"]
    t0 = time.perf_counter()
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs", random_state=seed
    )
    model.fit(X[tr], y[tr])
    train_seconds = time.perf_counter() - t0
    pv = model.predict_proba(X[va])[:, 1]
    pt = model.predict_proba(X[te])[:, 1]
    return _result_row(
        "logistic_regression", seed,
        metrics(y[va], pv)["pr_auc"],
        metrics(y[te], pt, threshold=best_threshold(y[va], pv)),
        bootstrap_ci(y[te], pt, "pr_auc", 500, seed),
        features=features, train_seconds=train_seconds,
        _test_y=y[te].tolist(), _test_proba=pt.tolist(),
    )


def run_mlp(df, features, y, masks, seed, epochs, hidden=96, lr=7e-4, dropout=.20):
    graph = build_temporal_graph(df, features, ALL_RELATIONS, history_k=1)
    tr, va, te = masks["train"], masks["validation"], masks["test"]
    model, history = train_mlp(
        graph, _mask_tensor(len(df), tr), _mask_tensor(len(df), va),
        epochs=epochs, seed=seed, hidden=hidden, lr=lr, dropout=dropout, patience=7
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(graph.x)).numpy()
    pv, pt = p[va], p[te]
    return _result_row(
        "feature_mlp", seed,
        metrics(y[va], pv)["pr_auc"],
        metrics(y[te], pt, threshold=best_threshold(y[va], pv)),
        bootstrap_ci(y[te], pt, "pr_auc", 500, seed),
        features=features, training_epochs=len(history), edges=int(graph.edge_index.size(1)),
        _test_y=y[te].tolist(), _test_proba=pt.tolist(),
    )


def run_gnn(
    df, features, y, masks, relations, history_k, seed, epochs,
    hidden=128, lr=7e-4, dropout=.15, patience=7,
):
    tr, va, te = masks["train"], masks["validation"], masks["test"]
    t0 = time.perf_counter()
    graph = build_temporal_graph(df, features, relations, history_k=history_k)
    build_seconds = time.perf_counter() - t0
    model, history = train_graphsage(
        graph, _mask_tensor(len(df), tr), _mask_tensor(len(df), va),
        epochs=epochs, seed=seed, hidden=hidden, lr=lr, dropout=dropout, patience=patience
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(graph.x, graph.edge_index)).numpy()
    pv, pt = p[va], p[te]
    return _result_row(
        "graphsage", seed,
        metrics(y[va], pv)["pr_auc"],
        metrics(y[te], pt, threshold=best_threshold(y[va], pv)),
        bootstrap_ci(y[te], pt, "pr_auc", 500, seed),
        relations=relations, history_k=history_k, features=features,
        training_epochs=len(history), edges=int(graph.edge_index.size(1)),
        graph_build_seconds=build_seconds,
        _test_y=y[te].tolist(), _test_proba=pt.tolist(),
    )


def flatten(rows):
    out = []
    for r in rows:
        if r["model"].startswith("paired_delta"):
            out.append({
                "model": r["model"], "seed": r.get("seed"),
                "validation_pr_auc": r.get("validation_pr_auc"),
                "test_pr_auc": r["test"]["pr_auc"],
                "delta_low": r["ci"].get("low"),
                "delta_high": r["ci"].get("high"),
                "p_b_gt_a": r["ci"].get("p_b_gt_a"),
            })
            continue
        test, ci = r["test"], r["ci"]
        out.append({
            "model": r["model"], "seed": r.get("seed"),
            "history_k": r.get("history_k"),
            "relations": ",".join(r.get("relations", [])),
            "feature_group": r.get("feature_group"),
            "validation_pr_auc": r.get("validation_pr_auc"),
            "test_pr_auc": test["pr_auc"],
            "test_roc_auc": test.get("roc_auc"),
            "ap_lift_vs_random": test.get("ap_lift_vs_random"),
            "precision_at_1pct": test.get("precision_at_1pct"),
            "recall_at_1pct": test.get("recall_at_1pct"),
            "brier": test.get("brier"),
            "ece_10": test.get("ece_10"),
            "ci_low": ci.get("low"),
            "ci_high": ci.get("high"),
            "train_seconds": r.get("train_seconds"),
            "graph_build_seconds": r.get("graph_build_seconds"),
            "edges": r.get("edges"),
            "training_epochs": r.get("training_epochs"),
        })
    return out


def paired_delta(a, b, name, seed):
    return {
        "model": f"paired_delta_{name}",
        "seed": seed,
        "validation_pr_auc": None,
        "test": {"pr_auc": float(b["test"]["pr_auc"] - a["test"]["pr_auc"])},
        "ci": paired_bootstrap_delta(
            np.asarray(a["_test_y"]), np.asarray(a["_test_proba"]),
            np.asarray(b["_test_proba"]), n_boot=1000, seed=seed,
        ),
    }


def run_core(df, features, y, masks, seed, epochs):
    groups = feature_sets(features)
    rows = [
        run_logistic(df, features, y, masks, seed),
        run_hgb(df, features, y, masks, seed),
        run_xgb(df, features, y, masks, seed),
        run_mlp(df, groups["all_features"], y, masks, seed, epochs),
        run_gnn(df, features, y, masks, ALL_RELATIONS, 3, seed, epochs),
    ]
    xgb = next(r for r in rows if r["model"] == "xgboost")
    gnn = next(r for r in rows if r["model"] == "graphsage")
    rows.append(paired_delta(xgb, gnn, "graphsage_minus_xgboost", seed))
    return rows


def main():
    p = argparse.ArgumentParser(description="Controlled ML research campaign for temporal fraud detection.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-rows", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--suite", choices=["core", "ablation", "robustness", "full"], default="core")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    p.add_argument("--history-values", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--output-dir", default="outputs/ml_study")
    args = p.parse_args()

    set_seed(args.seeds[0])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, features, y, masks = prepare(args.data_dir, args.max_rows)
    groups = feature_sets(features)
    rows = []

    # Core model ladder: linear -> boosted trees -> feature-only neural control -> graph model.
    rows.extend(run_core(df, features, y, masks, args.seeds[0], args.epochs))

    if args.suite in {"ablation", "full"}:
        # Feature contribution: remove all historical behavioral counts from both controls.
        for group_name in ["base_temporal_amount", "behavioral_history"]:
            fs = groups[group_name]
            rows.append(run_xgb(df, fs, y, masks, args.seeds[0]))
            rows[-1]["feature_group"] = group_name
            rows.append(run_mlp(df, fs, y, masks, args.seeds[0], args.epochs))
            rows[-1]["feature_group"] = group_name
            rows.append(run_gnn(df, fs, y, masks, ALL_RELATIONS, 3, args.seeds[0], args.epochs))
            rows[-1]["feature_group"] = group_name

        # History-depth sensitivity.
        for k in args.history_values:
            rows.append(run_gnn(df, features, y, masks, ALL_RELATIONS, k, args.seeds[0], args.epochs))

        # Relation-only and leave-one-out ablations.
        for relation in ALL_RELATIONS:
            rows.append(run_gnn(df, features, y, masks, [relation], 3, args.seeds[0], args.epochs))
            rows[-1]["ablation"] = "relation_only"
            rows.append(run_gnn(
                df, features, y, masks,
                [r for r in ALL_RELATIONS if r != relation],
                3, args.seeds[0], args.epochs
            ))
            rows[-1]["ablation"] = "leave_one_out"
            rows[-1]["removed_relation"] = relation

    if args.suite in {"robustness", "full"}:
        for seed in args.seeds[1:]:
            core_rows = run_core(df, features, y, masks, seed, args.epochs)
            rows.extend(core_rows)

    # Robustness summary by model across available seeds.
    clean = [r for r in rows if "_test_proba" in r and r["model"] in {
        "xgboost", "hist_gradient_boosting", "logistic_regression", "feature_mlp", "graphsage"
    }]
    summary = []
    for model_name in sorted(set(r["model"] for r in clean)):
        vals = [r["test"]["pr_auc"] for r in clean if r["model"] == model_name]
        if len(vals) > 1:
            summary.append({
                "model": model_name,
                "n_runs": len(vals),
                "mean_pr_auc": float(np.mean(vals)),
                "std_pr_auc": float(np.std(vals, ddof=1)),
                "median_pr_auc": float(np.median(vals)),
                "min_pr_auc": float(np.min(vals)),
                "max_pr_auc": float(np.max(vals)),
            })

    serializable = []
    for r in rows:
        q = dict(r)
        q.pop("_test_y", None)
        q.pop("_test_proba", None)
        serializable.append(q)

    payload = {
        "protocol": {
            "chronological_split": "70/15/15",
            "primary_metric": "average_precision",
            "test_threshold_selection": "validation_only",
            "bootstrap": 500,
            "paired_bootstrap": 1000,
            "seeds": args.seeds,
            "suite": args.suite,
        },
        "dataset": {"rows": len(df), "fraud_rate": float(y.mean()), "features": features},
        "feature_groups": groups,
        "results": serializable,
        "robustness_summary": summary,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(flatten(rows)).to_csv(out / "results.csv", index=False)

    print(json.dumps({
        "rows": len(rows),
        "output": str(out),
        "robustness_summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
