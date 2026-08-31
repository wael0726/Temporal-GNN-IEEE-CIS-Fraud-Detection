import numpy as np
import pandas as pd

from src.evaluate import paired_bootstrap_delta
from src.graph import build_temporal_graph


def test_retained_relation_counts_distinguish_raw_and_deduplicated_edges():
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
    assert sum(g.retained_relation_edge_counts.values()) == 2
    assert g.edge_time_delta_seconds.shape[0] == g.edge_index.shape[1]


def test_paired_bootstrap_reports_probability_of_positive_delta():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    a = np.array([.01, .05, .10, .20, .30, .40, .50, .60])
    b = np.array([.01, .02, .80, .70, .10, .90, .05, .20])
    result = paired_bootstrap_delta(y, a, b, n_boot=200, seed=42)
    assert "p_b_gt_a" in result
    assert 0.0 <= result["p_b_gt_a"] <= 1.0
    assert result["n_boot"] > 0
