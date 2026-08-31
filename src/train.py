from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .baseline import train_xgb
from .data import load_ieee_cis, load_ulb
from .evaluate import (
    apply_platt_scaler,
    best_threshold,
    bootstrap_ci,
    fit_platt_scaler,
    metrics,
)
from .features import build_temporal_features
from .graph import build_temporal_graph
from .graph_model import (
    HAS_PYG,
    predict_graphsage,
    train_graphsage,
    train_graphsage_sampled,
)

RELATIONS = ["card1", "addr1", "DeviceInfo", "P_emaildomain", "ProductCD"]


def _split(n: int):
    if n < 100:
        raise ValueError("Need at least 100 rows for a meaningful chronological split.")
    return int(n * 0.70), int(n * 0.85)


def _fit_transform(df, features, train_end):
    scaler = StandardScaler()
    arr = df[features].to_numpy(dtype=np.float32)
    scaler.fit(arr[:train_end])
    arr = scaler.transform(arr).astype(np.float32)
    out = df.copy()
    out.loc[:, features] = arr
    return out, scaler


def _save_plots(
    out: Path,
    df: pd.DataFrame,
    train_end: int,
    val_end: int,
    y_test,
    xgb_metrics,
    gnn_metrics,
    xgb_test_proba,
    gnn_test_proba,
):
    names = ["PR-AUC", "Recall", "F1"]
    xvals = [xgb_metrics["pr_auc"], xgb_metrics["recall"], xgb_metrics["f1"]]
    gvals = [gnn_metrics["pr_auc"], gnn_metrics["recall"], gnn_metrics["f1"]]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.18, xvals, 0.36, label="XGBoost")
    ax.bar(x + 0.18, gvals, 0.36, label="GraphSAGE")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Chronological test-set comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    xp, yp, _ = precision_recall_curve(y_test, xgb_test_proba)
    gp, gy, _ = precision_recall_curve(y_test, gnn_test_proba)
    ax.step(yp, xp, where="post", label=f"XGBoost (AP={xgb_metrics['pr_auc']:.3f})")
    ax.step(gy, gp, where="post", label=f"GraphSAGE (AP={gnn_metrics['pr_auc']:.3f})")
    ax.axhline(float(np.mean(y_test)), linestyle="--", linewidth=1, label="test prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall on chronological test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "precision_recall.png", dpi=160)
    plt.close(fig)

    temporal = df.copy()
    temporal["period"] = pd.to_datetime(temporal["timestamp"]).dt.floor("6h")
    rate = temporal.groupby("period", observed=True)["label"].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rate.index, rate.values)
    ax.axvline(
        pd.to_datetime(temporal["timestamp"].iloc[train_end]),
        linestyle="--",
        label="train/validation boundary",
    )
    ax.axvline(
        pd.to_datetime(temporal["timestamp"].iloc[val_end]),
        linestyle=":",
        label="validation/test boundary",
    )
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fraud_rate_over_time.png", dpi=160)
    plt.close(fig)


def _xgb_configs(tune: bool):
    if not tune:
        return [{}]
    return [
        {},
        {"max_depth": 4, "learning_rate": 0.06, "n_estimators": 450},
        {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 700, "min_child_weight": 3},
        {"max_depth": 7, "learning_rate": 0.035, "n_estimators": 700, "min_child_weight": 5},
        {"max_depth": 8, "learning_rate": 0.025, "n_estimators": 900, "min_child_weight": 8},
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 900, "subsample": 0.7, "colsample_bytree": 0.9},
    ]


def _gnn_configs(tune: bool):
    if not tune:
        return [(64, 1e-3, 0.25)]
    return [
        (64, 1e-3, 0.25),
        (96, 7e-4, 0.20),
        (128, 5e-4, 0.20),
        (64, 5e-4, 0.35),
        (96, 5e-4, 0.30),
        (128, 7e-4, 0.15),
    ]


def run(
    data_dir="data",
    dataset="ieee_cis",
    max_rows=100000,
    epochs=40,
    seed=42,
    history_k=3,
    tune=False,
    output_dir="artifacts",
    graph_mode="auto",
    sampled_batch_size=1024,
    neighbor_fanout=(20, 10),
    patience=7,
    bootstrap=300,
    auto_sampled_rows=75000,
    strict_calibration=False,
    compare_isotonic=False,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_pipeline = time.perf_counter()
    loader = load_ieee_cis if dataset == "ieee_cis" else load_ulb

    df = loader(data_dir, max_rows=max_rows)
    df, features = build_temporal_features(df)
    train_end, val_end = _split(len(df))
    df, scaler = _fit_transform(df, features, train_end)

    X = df[features].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    if y_train.sum() == 0 or y_val.sum() == 0 or y_test.sum() == 0:
        raise ValueError("Each chronological split must contain at least one fraud case.")
    # In strict mode, the validation window is divided chronologically:
    # the earlier portion is used for model selection and the later portion is
    # reserved for calibration/threshold selection. The final test remains untouched.
    if strict_calibration:
        val_selection_end = train_end + max(1, int(0.70 * (val_end - train_end)))
        selection_slice = slice(train_end, val_selection_end)
        calibration_slice = slice(val_selection_end, val_end)
        y_val_selection = y[selection_slice]
        y_calibration = y[calibration_slice]
        if y_val_selection.sum() == 0 or y_calibration.sum() == 0:
            raise ValueError("Strict calibration requires positives in both validation sub-windows.")
    else:
        val_selection_end = val_end
        selection_slice = slice(train_end, val_end)
        calibration_slice = slice(train_end, val_end)
        y_val_selection = y_val
        y_calibration = y_val

    # -------------------- XGBoost baseline --------------------
    best_xgb, best_xgb_score, best_xgb_cfg = None, -1.0, None
    xgb_tuning = []
    for cfg in _xgb_configs(tune):
        model = train_xgb(X[:train_end], y_train, seed=seed, params=cfg)
        val_proba = model.predict_proba(X[selection_slice])[:, 1]
        score = metrics(y_val_selection, val_proba)["pr_auc"]
        xgb_tuning.append({"params": cfg, "val_pr_auc": score})
        if score > best_xgb_score:
            best_xgb, best_xgb_score, best_xgb_cfg = model, score, cfg

    xgb_val_proba = best_xgb.predict_proba(X[calibration_slice])[:, 1]
    xgb_test_proba = best_xgb.predict_proba(X[val_end:])[:, 1]
    xgb_threshold = best_threshold(y_calibration, xgb_val_proba)
    xgb_metrics = metrics(y_test, xgb_test_proba, threshold=xgb_threshold)

    # Platt calibration is fit on validation predictions and evaluated on test.
    xgb_cal = fit_platt_scaler(y_calibration, xgb_val_proba)
    xgb_test_cal = apply_platt_scaler(xgb_cal, xgb_test_proba)
    xgb_calibrated_metrics = metrics(
        y_test, xgb_test_cal, threshold=best_threshold(y_calibration, apply_platt_scaler(xgb_cal, xgb_val_proba))
    )

    # -------------------- Temporal graph --------------------
    graph = build_temporal_graph(df, features, RELATIONS, history_k=history_k)
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    val_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_selection_end] = True

    requested_mode = graph_mode
    if graph_mode == "auto":
        graph_mode = "sampled" if len(df) >= auto_sampled_rows and HAS_PYG else "full"
    if graph_mode == "sampled" and not HAS_PYG:
        raise RuntimeError(
            "graph_mode=sampled requires torch-geometric. Install the project requirements "
            "or use --graph-mode full."
        )

    best_gnn, best_gnn_score, best_gnn_cfg, best_history = None, -1.0, None, []
    tuning_rows = []
    selection_val_indices = np.arange(train_end, val_selection_end, dtype=np.int64)
    calibration_start = val_selection_end if strict_calibration else train_end
    calibration_val_indices = np.arange(calibration_start, val_end, dtype=np.int64)
    for hidden, lr, dropout in _gnn_configs(tune):
        if graph_mode == "sampled":
            model, history, _ = train_graphsage_sampled(
                graph,
                np.arange(train_end, dtype=np.int64),
                selection_val_indices,
                epochs=epochs,
                lr=lr,
                hidden=hidden,
                dropout=dropout,
                seed=seed,
                patience=patience,
                batch_size=sampled_batch_size,
                num_neighbors=neighbor_fanout,
            )
            val_proba = predict_graphsage(
                model,
                graph,
                selection_val_indices,
                sampled=True,
                batch_size=sampled_batch_size,
                num_neighbors=neighbor_fanout,
            )
        else:
            model, history = train_graphsage(
                graph,
                train_mask,
                val_mask,
                epochs=epochs,
                lr=lr,
                hidden=hidden,
                dropout=dropout,
                seed=seed,
                patience=patience,
            )
            val_proba = predict_graphsage(
                model, graph, selection_val_indices
            )

        score = metrics(y_val_selection, val_proba)["pr_auc"]
        tuning_rows.append(
            {"hidden": hidden, "lr": lr, "dropout": dropout, "val_pr_auc": score}
        )
        if score > best_gnn_score:
            best_gnn, best_gnn_score = model, score
            best_gnn_cfg = {"hidden": hidden, "lr": lr, "dropout": dropout}
            best_history = history

    val_indices = calibration_val_indices
    test_indices = np.arange(val_end, len(df), dtype=np.int64)
    gnn_val_proba = predict_graphsage(
        best_gnn, graph, val_indices, sampled=(graph_mode == "sampled"),
        batch_size=sampled_batch_size, num_neighbors=neighbor_fanout
    )
    gnn_test_proba = predict_graphsage(
        best_gnn, graph, test_indices, sampled=(graph_mode == "sampled"),
        batch_size=sampled_batch_size, num_neighbors=neighbor_fanout
    )
    gnn_threshold = best_threshold(y_calibration, gnn_val_proba)
    gnn_metrics = metrics(y_test, gnn_test_proba, threshold=gnn_threshold)

    gnn_cal = fit_platt_scaler(y_calibration, gnn_val_proba)
    gnn_test_cal = apply_platt_scaler(gnn_cal, gnn_test_proba)
    gnn_val_cal = apply_platt_scaler(gnn_cal, gnn_val_proba)
    gnn_calibrated_metrics = metrics(
        y_test, gnn_test_cal, threshold=best_threshold(y_calibration, gnn_val_cal)
    )

    calibration_comparison = {}
    if compare_isotonic:
        from .evaluate import apply_isotonic_scaler, fit_isotonic_scaler
        if len(y_calibration) >= 1000 and len(np.unique(y_calibration)) >= 2:
            xgb_iso = fit_isotonic_scaler(y_calibration, xgb_val_proba)
            gnn_iso = fit_isotonic_scaler(y_calibration, gnn_val_proba)
            xgb_iso_test = apply_isotonic_scaler(xgb_iso, xgb_test_proba)
            gnn_iso_test = apply_isotonic_scaler(gnn_iso, gnn_test_proba)
            calibration_comparison = {
                "xgboost_isotonic": metrics(
                    y_test, xgb_iso_test,
                    threshold=best_threshold(y_calibration, apply_isotonic_scaler(xgb_iso, xgb_val_proba))
                ),
                "graphsage_isotonic": metrics(
                    y_test, gnn_iso_test,
                    threshold=best_threshold(y_calibration, apply_isotonic_scaler(gnn_iso, gnn_val_proba))
                ),
            }
        else:
            calibration_comparison = {
                "isotonic": {
                    "status": "skipped",
                    "reason": "requires >=1000 calibration samples and both classes",
                    "calibration_rows": int(len(y_calibration)),
                }
            }

    relation_counts = graph.relation_edge_counts
    results = {
        "dataset": dataset,
        "rows_used": len(df),
        "fraud_count": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "split": {
            "train": train_end,
            "validation": val_end - train_end,
            "test": len(df) - val_end,
            "train_start": str(df["timestamp"].iloc[0]),
            "train_end": str(df["timestamp"].iloc[train_end - 1]),
            "validation_end": str(df["timestamp"].iloc[val_end - 1]),
            "test_end": str(df["timestamp"].iloc[-1]),
        },
        "features": features,
        "feature_engineering": {
            "causal": True,
            "missing_values_form_entities": False,
            "scaler_fit_on": "train_only",
        },
        "graph": {
            "relation_columns": RELATIONS,
            "history_k": history_k,
            "nodes": len(df),
            "edges": int(graph.edge_index.size(1)),
            "edges_by_relation": relation_counts,
            "retained_edges_by_relation": graph.retained_relation_edge_counts,
            "causal_direction": "past_to_current",
            "deduplicated_edges": True,
            "missing_values_form_entities": False,
        },
        "xgboost": xgb_metrics,
        "xgboost_calibrated": xgb_calibrated_metrics,
        "graphsage": gnn_metrics,
        "graphsage_calibrated": gnn_calibrated_metrics,
        "xgboost_validation_pr_auc": best_xgb_score,
        "graphsage_validation_pr_auc": best_gnn_score,
        "selected_configs": {
            "xgboost": best_xgb_cfg,
            "graphsage": best_gnn_cfg,
        },
        "tuning": {"xgboost": xgb_tuning, "graphsage": tuning_rows},
        "graphsage_training": {
            "epochs_requested": epochs,
            "epochs_completed": len(best_history),
            "patience": patience,
            "history": best_history,
        },
        "scalability": {
            "requested_graph_mode": requested_mode,
            "effective_graph_mode": graph_mode,
            "auto_sampled_rows": auto_sampled_rows,
            "pipeline_seconds": time.perf_counter() - t_pipeline,
            "sampled_batch_size": sampled_batch_size,
            "neighbor_fanout": list(neighbor_fanout),
            "torch_geometric_available": HAS_PYG,
        },
        "confidence_intervals": {
            "xgboost_pr_auc": bootstrap_ci(y_test, xgb_test_proba, "pr_auc", bootstrap, seed),
            "graphsage_pr_auc": bootstrap_ci(y_test, gnn_test_proba, "pr_auc", bootstrap, seed),
        },
        "calibration_comparison": calibration_comparison,
        "protocol": {
            "strict_calibration": strict_calibration,
            "selection_validation_rows": int(val_selection_end - train_end),
            "calibration_rows": int(val_end - val_selection_end),
            "test_rows": int(len(df) - val_end),
        },
        "reproducibility": {
            "seed": seed,
            "max_rows": max_rows,
            "epochs": epochs,
            "history_k": history_k,
        },
        "primary_metric": "average_precision",
        "random_baseline_pr_auc": float(y_test.mean()),
    }

    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    joblib.dump(scaler, out / "feature_scaler.joblib")
    best_xgb.save_model(out / "xgboost.json")
    torch.save(
        {
            "state_dict": best_gnn.state_dict(),
            "features": features,
            "history_k": history_k,
            "relations": RELATIONS,
            "hidden": best_gnn_cfg["hidden"],
            "dropout": best_gnn_cfg["dropout"],
            "lr": best_gnn_cfg["lr"],
            "graph_mode": graph_mode,
        },
        out / "graphsage.pt",
    )
    joblib.dump(xgb_cal, out / "xgboost_platt_calibrator.joblib")
    joblib.dump(gnn_cal, out / "graphsage_platt_calibrator.joblib")

    pred = pd.DataFrame(
        {
            "row_id": df["row_id"].iloc[val_end:].to_numpy(),
            "transaction_id": (
                df["TransactionID"].iloc[val_end:].to_numpy()
                if "TransactionID" in df
                else df["row_id"].iloc[val_end:].to_numpy()
            ),
            "timestamp": df["timestamp"].iloc[val_end:].astype(str).to_numpy(),
            "label": y_test,
            "xgb_probability": xgb_test_proba,
            "xgb_probability_calibrated": xgb_test_cal,
            "graphsage_probability": gnn_test_proba,
            "graphsage_probability_calibrated": gnn_test_cal,
        }
    )
    pred.to_csv(out / "predictions.csv", index=False)
    _save_plots(
        out, df, train_end, val_end, y_test,
        xgb_metrics, gnn_metrics, xgb_test_proba, gnn_test_proba
    )
    return results


def main():
    p = argparse.ArgumentParser(
        description="Rigorous chronological XGBoost vs causal GraphSAGE experiment."
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", choices=["ieee_cis", "ulb"], default="ieee_cis")
    p.add_argument("--max-rows", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--history-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--output-dir", default="artifacts")
    p.add_argument("--graph-mode", choices=["auto", "full", "sampled"], default="auto")
    p.add_argument("--sampled-batch-size", type=int, default=1024)
    p.add_argument("--neighbor-fanout", type=int, nargs=2, default=(20, 10))
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--auto-sampled-rows", type=int, default=75000)
    p.add_argument("--strict-calibration", action="store_true")
    p.add_argument("--compare-isotonic", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
