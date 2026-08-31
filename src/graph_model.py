from __future__ import annotations

import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.nn import SAGEConv
    HAS_PYG = True
except ImportError:
    Data = None
    NeighborLoader = None
    HAS_PYG = False

    class SAGEConv(nn.Module):
        """Dependency-free fallback used for local tests and CPU-only demos."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.lin = nn.Linear(in_channels * 2, out_channels)

        def forward(self, x, edge_index):
            src, dst = edge_index
            agg = torch.zeros_like(x)
            if src.numel():
                agg.index_add_(0, dst, x[src])
                deg = torch.zeros(x.size(0), device=x.device)
                deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
                agg = agg / deg.clamp_min(1).unsqueeze(1)
            return self.lin(torch.cat([x, agg], dim=1))


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.classifier = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return self.classifier(h).squeeze(-1)



class FeatureMLP(nn.Module):
    """Graph-free control using the same node features and hidden capacity."""
    def __init__(self, in_channels: int, hidden: int = 64, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    graph,
    train_mask,
    val_mask,
    epochs=40,
    lr=1e-3,
    hidden=64,
    dropout=0.25,
    seed=42,
    patience=7,
):
    """Train a graph-free feature-only control with the same causal inputs."""
    set_seed(seed)
    model = FeatureMLP(graph.x.size(1), hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = graph.labels[train_mask].sum().item()
    neg = train_mask.sum().item() - pos
    if pos <= 0:
        raise ValueError("Training split contains no positive examples.")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
    )
    best_state = None
    best_val_ap = -float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.x)
        loss = criterion(logits[train_mask], graph.labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(graph.x)[val_mask]).cpu().numpy()
        ap = float(average_precision_score(graph.labels[val_mask].cpu().numpy(), p))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": ap})
        if ap > best_val_ap + 1e-6:
            best_val_ap = ap
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _full_proba(model, graph):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(graph.x, graph.edge_index)).cpu().numpy()


def train_graphsage(
    graph,
    train_mask,
    val_mask,
    epochs=40,
    lr=1e-3,
    hidden=64,
    dropout=0.25,
    seed=42,
    patience=7,
    gradient_clip=5.0,
):
    """Full-batch GraphSAGE with validation-AP early stopping.

    This is the reference implementation: deterministic in structure, simple
    enough for ablations, and appropriate for graphs that fit comfortably in
    memory.
    """
    set_seed(seed)
    model = GraphSAGE(graph.x.size(1), hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = graph.labels[train_mask].sum().item()
    neg = train_mask.sum().item() - pos
    if pos <= 0:
        raise ValueError("Training split contains no positive examples.")
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = None
    best_val_ap = -float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.edge_index)
        loss = criterion(logits[train_mask], graph.labels[train_mask])
        loss.backward()
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()

        val_proba = _full_proba(model, graph)[val_mask.cpu().numpy()]
        val_ap = float(average_precision_score(
            graph.labels[val_mask].cpu().numpy(), val_proba
        ))
        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "val_pr_auc": val_ap,
        })

        if val_ap > best_val_ap + 1e-6:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _make_pyg_data(graph):
    if not HAS_PYG:
        raise RuntimeError("PyTorch Geometric is not installed.")
    return Data(x=graph.x, edge_index=graph.edge_index, y=graph.labels)


def _sampled_predict(model, data, node_idx, num_neighbors=(20, 10), batch_size=1024):
    loader = NeighborLoader(
        data,
        input_nodes=node_idx,
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=False,
    )
    result = np.empty(len(node_idx), dtype=np.float32)
    for batch in loader:
        batch = batch.to(data.x.device)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            p = torch.sigmoid(logits).cpu().numpy()
        global_ids = batch.n_id[:batch.batch_size].cpu().numpy()
        # node_idx is sorted by construction in our chronological pipeline.
        result[np.searchsorted(node_idx, global_ids)] = p
    return result


def train_graphsage_sampled(
    graph,
    train_indices,
    val_indices,
    epochs=30,
    lr=1e-3,
    hidden=64,
    dropout=0.25,
    seed=42,
    patience=5,
    batch_size=1024,
    num_neighbors=(20, 10),
    gradient_clip=5.0,
):
    """Mini-batch GraphSAGE for larger graphs using PyG NeighborLoader.

    Only seed nodes contribute to the loss. The sampled subgraph still contains
    historical context, while memory use is bounded by the neighborhood budget.
    """
    if not HAS_PYG:
        raise RuntimeError("Sampled GraphSAGE requires torch-geometric.")
    set_seed(seed)
    data = _make_pyg_data(graph)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)

    model = GraphSAGE(graph.x.size(1), hidden=hidden, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    train_y = graph.labels[train_indices]
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    if pos <= 0:
        raise ValueError("Training split contains no positive examples.")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / pos], dtype=torch.float32)
    )

    loader = NeighborLoader(
        data,
        input_nodes=torch.as_tensor(train_indices, dtype=torch.long),
        num_neighbors=list(num_neighbors),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_val_ap = -float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size]
            loss = criterion(logits, y)
            loss.backward()
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        model.eval()
        val_proba = _sampled_predict(
            model, data, val_indices, num_neighbors=num_neighbors, batch_size=batch_size
        )
        val_ap = float(average_precision_score(graph.labels[val_indices].numpy(), val_proba))
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "val_pr_auc": val_ap,
        })
        if val_ap > best_val_ap + 1e-6:
            best_val_ap = val_ap
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, data


def predict_graphsage(model, graph, node_indices=None, sampled=False, batch_size=1024, num_neighbors=(20, 10)):
    """Predict probabilities for selected global node indices."""
    if node_indices is None:
        node_indices = np.arange(len(graph.x), dtype=np.int64)
    node_indices = np.asarray(node_indices, dtype=np.int64)
    if sampled:
        data = _make_pyg_data(graph)
        return _sampled_predict(model, data, node_indices, num_neighbors, batch_size)
    return _full_proba(model, graph)[node_indices]
