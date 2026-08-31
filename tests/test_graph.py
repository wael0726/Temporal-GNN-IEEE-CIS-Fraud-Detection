import pandas as pd
import numpy as np
from src.graph import build_temporal_graph


def test_graph_has_only_past_to_current_relation_edges():
    df = pd.DataFrame({
        "row_id": [0, 1, 2], "label": [0, 1, 0],
        "card1": ["A", "A", "A"],
        "timestamp": pd.date_range("2020-01-01", periods=3, freq="min"), "x": [0.0, 1.0, 2.0],
    })
    g = build_temporal_graph(df, ["x"], ["card1"], history_k=1)
    src, dst = g.edge_index.numpy()
    assert np.all(src < dst)


def test_graph_contains_no_self_loops():
    df = pd.DataFrame({"row_id": [0, 1], "label": [0, 1], "card1": ["A", "A"], "x": [1.0, 2.0], "timestamp": pd.date_range("2020-01-01", periods=2, freq="min")})
    g = build_temporal_graph(df, ["x"], ["card1"], history_k=2)
    src, dst = g.edge_index.numpy()
    assert not np.any(src == dst)
