from __future__ import annotations

import xgboost as xgb


def train_xgb(X, y, seed=42, params=None):
    """Train a strong, imbalance-aware tabular baseline."""
    p = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=4,
        scale_pos_weight=max(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0), 1.0),
    )
    if params:
        p.update(params)
    model = xgb.XGBClassifier(**p)
    model.fit(X, y)
    return model
