import numpy as np
import pandas as pd
import torch

from src.evaluate import best_threshold, metrics
from src.features import build_temporal_features
from src.graph import build_temporal_graph


def _df():
    return pd.DataFrame({
        "row_id": range(5),
        "label": [0, 1, 0, 1, 0],
        "timestamp": pd.date_range("2020-01-01", periods=5, freq="min"),
        "TransactionDT": np.arange(5) * 60,
        "amount": [10., 20., 30., 40., 50.],
        "card1": ["A", "A", None, "A", "B"],
        "addr1": ["X", None, "X", "X", "Y"],
        "DeviceInfo": [None, None, "D", "D", None],
        "P_emaildomain": ["e", "e", None, "e", "f"],
        "ProductCD": ["W", "W", "W", "W", "H"],
    })


def test_temporal_features_are_causal_and_missing_is_not_an_entity():
    out, features = build_temporal_features(_df())
    assert out.loc[0, "card1_past_count"] == 0
    assert out.loc[1, "card1_past_count"] == 1
    assert out.loc[2, "card1_past_count"] == 0
    assert out.loc[1, "DeviceInfo_past_count"] == 0
    assert "week_cycle_sin" in features
    assert "day_of_week" not in features


def test_graph_has_only_past_to_current_edges_and_no_missing_entity_edges():
    df, features = build_temporal_features(_df())
    graph = build_temporal_graph(df, features, ["card1", "addr1", "DeviceInfo"], history_k=3)
    src, dst = graph.edge_index.numpy()
    assert np.all(src < dst)
    assert not np.any(src == dst)
    # Missing card1/device values never connect to one another.
    assert all(not (df.loc[s, "card1"] is None and df.loc[d, "card1"] is None) for s, d in zip(src, dst))


def test_graph_deduplicates_multi_relation_edges():
    df = _df().copy()
    df["addr1"] = df["card1"]
    graph = build_temporal_graph(df, ["amount"], ["card1", "addr1"], history_k=1)
    pairs = list(zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist()))
    assert len(pairs) == len(set(pairs))


def test_threshold_is_selected_on_validation_and_reused_on_test():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    p_val = np.array([0.05, 0.20, 0.45, 0.55, 0.80, 0.95])
    threshold = best_threshold(y_val, p_val)
    y_test = np.array([0, 1, 0, 1])
    p_test = np.array([0.40, 0.60, 0.90, 0.95])
    result = metrics(y_test, p_test, threshold=threshold)
    assert 0.0 < threshold < 1.0
    assert result["threshold"] == threshold
    assert 0.0 <= result["precision_at_1pct"] <= 1.0
    assert 0.0 <= result["recall_at_1pct"] <= 1.0
