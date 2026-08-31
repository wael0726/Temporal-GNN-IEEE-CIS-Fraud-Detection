import numpy as np
import pandas as pd

from src.evaluate import bootstrap_ci, fit_platt_scaler, apply_platt_scaler, metrics
from src.graph import build_temporal_graph


def test_evaluation_reports_random_lift_and_operational_budgets():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1])
    p = np.array([0.01, 0.02, 0.03, 0.90, 0.80, 0.04, 0.05, 0.70])
    result = metrics(y, p, threshold=0.5)
    assert result["ap_lift_vs_random"] > 1.0
    assert "precision_at_0.1pct" in result
    assert "recall_at_1pct" in result


def test_platt_calibration_is_fitted_on_validation_and_returns_probabilities():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    p = np.array([0.05, 0.10, 0.30, 0.60, 0.20, 0.80, 0.15, 0.90])
    cal = fit_platt_scaler(y, p)
    calibrated = apply_platt_scaler(cal, p)
    assert np.all(calibrated >= 0.0)
    assert np.all(calibrated <= 1.0)
    assert len(calibrated) == len(p)


def test_graph_relation_counts_include_pre_deduplication_evidence():
    df = pd.DataFrame({
        "row_id": [0, 1, 2],
        "label": [0, 1, 0],
        "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
        "card1": ["A", "A", "A"],
        "addr1": ["A", "A", "A"],
        "x": [0.0, 1.0, 2.0],
    })
    g = build_temporal_graph(df, ["x"], ["card1", "addr1"], history_k=1)
    assert g.relation_edge_counts["card1"] == 2
    assert g.relation_edge_counts["addr1"] == 2
    assert g.edge_index.size(1) == 2  # same pairs are deduplicated in the model graph
