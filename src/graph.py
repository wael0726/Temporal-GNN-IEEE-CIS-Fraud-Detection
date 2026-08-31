from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import torch


@dataclass
class GraphBundle:
    x: torch.Tensor
    edge_index: torch.Tensor
    labels: torch.Tensor
    row_ids: np.ndarray
    edge_types: list[str]
    relation_edge_counts: dict[str, int]
    retained_relation_edge_counts: dict[str, int]
    edge_time_delta_seconds: torch.Tensor


def _append_temporal_edges(
    df: pd.DataFrame, columns: list[str], history_k: int
) -> tuple[list[tuple[int, int]], list[str], dict[str, int], dict[str, int]]:
    if history_k < 1:
        raise ValueError("history_k must be >= 1")
    edges: list[tuple[int, int]] = []
    edge_types: list[str] = []
    seen: set[tuple[int, int]] = set()
    relation_edge_counts = {col: 0 for col in columns}
    retained_relation_edge_counts = {col: 0 for col in columns}

    for col in columns:
        if col not in df.columns:
            continue
        history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=history_k))
        values = df[col].astype("string")
        for idx, value in enumerate(values):
            if pd.isna(value):
                continue
            key = str(value)
            for prev_idx in history[key]:
                relation_edge_counts[col] += 1
                pair = (prev_idx, idx)
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)
                    edge_types.append(col)
                    retained_relation_edge_counts[col] += 1
            history[key].append(idx)

    return edges, edge_types, relation_edge_counts, retained_relation_edge_counts


def build_temporal_graph(
    df: pd.DataFrame,
    feature_columns: list[str],
    relation_columns: list[str],
    history_k: int = 3,
    add_global_temporal_edges: bool = False,
) -> GraphBundle:
    """Build a deduplicated directed graph where every edge is past -> current.

    Missing categorical values are skipped and therefore never create a shared
    synthetic entity. Edge time deltas are retained for auditing and future
    time-aware message-passing variants.
    """
    if not pd.to_datetime(df["timestamp"]).is_monotonic_increasing:
        raise ValueError("Graph construction requires chronologically sorted input.")

    edges, edge_types, relation_edge_counts, retained_relation_edge_counts = _append_temporal_edges(
        df, relation_columns, history_k
    )
    if add_global_temporal_edges:
        for i in range(1, len(df)):
            edges.append((i - 1, i))
            edge_types.append("global_temporal")
        relation_edge_counts["global_temporal"] = len(df) - 1
        retained_relation_edge_counts["global_temporal"] = len(df) - 1

    if edges:
        edge_array = np.asarray(edges, dtype=np.int64)
        edge_index = torch.tensor(edge_array.T, dtype=torch.long)
        timestamps = pd.to_datetime(df["timestamp"]).astype("int64").to_numpy() // 10**9
        deltas = timestamps[edge_array[:, 1]] - timestamps[edge_array[:, 0]]
        edge_time_delta_seconds = torch.tensor(deltas, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_time_delta_seconds = torch.empty((0,), dtype=torch.float32)

    x = torch.tensor(df[feature_columns].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    return GraphBundle(
        x=x,
        edge_index=edge_index,
        labels=y,
        row_ids=df["row_id"].to_numpy(),
        edge_types=edge_types,
        relation_edge_counts=relation_edge_counts,
        retained_relation_edge_counts=retained_relation_edge_counts,
        edge_time_delta_seconds=edge_time_delta_seconds,
    )
