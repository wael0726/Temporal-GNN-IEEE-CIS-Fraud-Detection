from __future__ import annotations

from pathlib import Path
import json

from fastapi import FastAPI

app = FastAPI(title="Temporal Fraud GNN API", version="1.0.0")
ARTIFACTS = Path("artifacts")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return {"status": "not_trained", "message": "Run the training pipeline first."}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/model-info")
def model_info():
    path = ARTIFACTS / "metrics.json"
    if not path.exists():
        return {"status": "not_trained"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": payload.get("dataset"),
        "rows_used": payload.get("rows_used"),
        "graph": payload.get("graph"),
        "features": payload.get("features"),
        "models": ["XGBoost", "GraphSAGE"],
    }
