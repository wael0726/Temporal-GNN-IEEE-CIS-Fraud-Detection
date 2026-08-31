from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ALERT_RATES = (0.001, 0.005, 0.01, 0.02, 0.05)


def best_threshold(y_true, proba) -> float:
    """Select an operating threshold on validation data only."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    thresholds = np.unique(np.clip(proba, 0.0, 1.0))
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    if thresholds.size == 0:
        return 0.5
    scores = np.array(
        [f1_score(y_true, proba >= t, zero_division=0) for t in thresholds]
    )
    best = np.flatnonzero(scores == scores.max())
    return float(thresholds[best[-1]])


def _top_k(y_true, proba, fraction: float) -> int:
    return max(1, min(len(y_true), int(np.ceil(len(y_true) * fraction))))


def _precision_at_k(y_true, proba, k: int) -> float:
    if len(y_true) == 0:
        return 0.0
    k = min(max(int(k), 1), len(y_true))
    idx = np.argsort(-np.asarray(proba), kind="stable")[:k]
    return float(np.asarray(y_true)[idx].mean())


def _recall_at_k(y_true, proba, k: int) -> float:
    y_true = np.asarray(y_true)
    positives = int(y_true.sum())
    if positives == 0 or len(y_true) == 0:
        return 0.0
    k = min(max(int(k), 1), len(y_true))
    idx = np.argsort(-np.asarray(proba), kind="stable")[:k]
    return float(y_true[idx].sum() / positives)


def ranking_metrics(y_true, proba) -> dict:
    """Threshold-free ranking metrics plus operational top-k metrics."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    prevalence = float(y_true.mean()) if len(y_true) else 0.0
    ap = float(average_precision_score(y_true, proba)) if len(y_true) else 0.0
    out = {
        "pr_auc": ap,
        "roc_auc": float(roc_auc_score(y_true, proba))
        if len(np.unique(y_true)) > 1 else 0.5,
        "prevalence": prevalence,
        "random_baseline_pr_auc": prevalence,
        "ap_lift_vs_random": float(ap / prevalence) if prevalence > 0 else 0.0,
    }
    for rate in ALERT_RATES:
        pct = f"{rate * 100:g}pct"
        k = _top_k(y_true, proba, rate)
        out[f"precision_at_{pct}"] = _precision_at_k(y_true, proba, k)
        out[f"recall_at_{pct}"] = _recall_at_k(y_true, proba, k)
        out[f"alerts_at_{pct}"] = int(k)
    return out


def fit_platt_scaler(y_val, val_proba):
    """Fit monotonic sigmoid calibration on an independent validation/calibration set."""
    y_val = np.asarray(y_val, dtype=int)
    val_proba = np.clip(np.asarray(val_proba, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y_val)) < 2:
        raise ValueError("Calibration requires both classes in the calibration set.")
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(np.log(val_proba / (1.0 - val_proba)).reshape(-1, 1), y_val)
    return model


def apply_platt_scaler(calibrator, proba):
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(proba / (1.0 - proba)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def fit_isotonic_scaler(y_val, val_proba):
    """Fit non-parametric monotonic calibration.

    Isotonic calibration can overfit small calibration sets; use it only when
    the calibration window contains enough positive and negative examples.
    """
    y_val = np.asarray(y_val, dtype=int)
    val_proba = np.asarray(val_proba, dtype=float)
    if len(y_val) < 1000 or len(np.unique(y_val)) < 2:
        raise ValueError("Isotonic calibration requires >=1000 calibration samples and both classes.")
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(val_proba, y_val)
    return model


def apply_isotonic_scaler(calibrator, proba):
    return np.asarray(calibrator.predict(np.asarray(proba, dtype=float)), dtype=float)


def calibration_metrics(y_true, proba) -> dict:
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    if len(y_true) == 0:
        return {"brier": 0.0, "log_loss": 0.0, "ece_10": 0.0}
    frac_pos, _ = calibration_curve(
        y_true, proba, n_bins=10, strategy="quantile"
    )
    order = np.argsort(proba, kind="stable")
    chunks = np.array_split(order, min(10, len(order)))
    ece = 0.0
    for idx in chunks:
        if len(idx) == 0:
            continue
        ece += (len(idx) / len(y_true)) * abs(
            float(y_true[idx].mean()) - float(proba[idx].mean())
        )
    return {
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "ece_10": float(ece),
        "calibration_bins": int(len(frac_pos)),
    }


def metrics(y_true, proba, threshold: float | None = None) -> dict:
    """Complete evaluation report.

    Thresholds must be selected on validation/calibration data before final
    test reporting. Passing an explicit threshold is therefore the safe default
    for held-out test sets.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) != len(proba):
        raise ValueError("y_true and proba must have the same length")
    threshold = best_threshold(y_true, proba) if threshold is None else float(threshold)
    pred = (proba >= threshold).astype(int)
    out = ranking_metrics(y_true, proba)
    out.update(calibration_metrics(y_true, proba))
    out.update(
        {
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "threshold": threshold,
        }
    )
    return out


def bootstrap_ci(y_true, proba, metric="pr_auc", n_boot=1000, seed=42) -> dict:
    """Non-parametric bootstrap confidence interval on a held-out set."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(y_true) < 2 or n_boot <= 0:
        return {"estimate": 0.0, "low": 0.0, "high": 0.0, "n_boot": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yy, pp = y_true[idx], proba[idx]
        if metric == "pr_auc":
            values.append(average_precision_score(yy, pp))
        elif metric == "roc_auc":
            if len(np.unique(yy)) < 2:
                continue
            values.append(roc_auc_score(yy, pp))
        else:
            raise ValueError("Supported bootstrap metrics: pr_auc, roc_auc")
    if not values:
        return {
            "estimate": float(ranking_metrics(y_true, proba)[metric]),
            "low": None,
            "high": None,
            "n_boot": 0,
        }
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "estimate": float(ranking_metrics(y_true, proba)[metric]),
        "low": float(lo),
        "high": float(hi),
        "n_boot": len(values),
    }


def paired_bootstrap_delta(y_true, proba_a, proba_b, n_boot=1000, seed=42) -> dict:
    """Paired bootstrap CI and one-sided probability that AP(B) > AP(A).

    The same resampled transactions are used for both models, which is more
    informative than comparing two independent confidence intervals.
    """
    y_true = np.asarray(y_true)
    a = np.asarray(proba_a, dtype=float)
    b = np.asarray(proba_b, dtype=float)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("All paired arrays must have the same length.")
    if len(y_true) < 2:
        return {"delta_ap": 0.0, "low": None, "high": None, "p_b_gt_a": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        yy = y_true[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(
            average_precision_score(yy, b[idx])
            - average_precision_score(yy, a[idx])
        )
    observed = average_precision_score(y_true, b) - average_precision_score(y_true, a)
    if not deltas:
        return {"delta_ap": float(observed), "low": None, "high": None, "p_b_gt_a": None, "n_boot": 0}
    low, high = np.percentile(deltas, [2.5, 97.5])
    deltas = np.asarray(deltas)
    return {
        "delta_ap": float(observed),
        "low": float(low),
        "high": float(high),
        "p_b_gt_a": float(np.mean(deltas > 0.0)),
        "n_boot": int(len(deltas)),
    }
