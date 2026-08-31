"""Generate additional evaluation figures from an existing artifacts/predictions.csv."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ART = Path("artifacts")

def main():
    metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(ART / "predictions.csv")
    y = pred["label"].to_numpy()
    for model, col, threshold in [
        ("XGBoost", "xgb_probability", metrics["xgboost"]["threshold"]),
        ("GraphSAGE", "graphsage_probability", metrics["graphsage"]["threshold"]),
    ]:
        p = pred[col].to_numpy()
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ConfusionMatrixDisplay.from_predictions(y, p >= threshold, labels=[0, 1], ax=ax, values_format="d")
        ax.set_title(f"{model} — chronological test")
        fig.tight_layout()
        fig.savefig(ART / f"confusion_{model.lower()}.png", dpi=160)
        plt.close(fig)

    # Compact text summary for quick inspection in a terminal.
    print("Rows:", metrics["rows_used"])
    print("Fraud rate:", f"{metrics['fraud_rate']:.4%}")
    for model in ("xgboost", "graphsage"):
        m = metrics[model]
        print(f"{model}: PR-AUC={m['pr_auc']:.6f} ROC-AUC={m['roc_auc']:.6f} F1={m['f1']:.6f}")

if __name__ == "__main__":
    main()
