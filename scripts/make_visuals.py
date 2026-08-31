from pathlib import Path
import matplotlib.pyplot as plt

OUT = Path("artifacts")
OUT.mkdir(exist_ok=True)

# 1) Architecture figure
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.axis("off")
steps = [
    ("IEEE-CIS\ntransactions", 0.08),
    ("Causal temporal\nfeatures", 0.27),
    ("Chronological\nsplit", 0.46),
    ("XGBoost\nbaseline", 0.65),
    ("Past → current graph\n+ GraphSAGE", 0.86),
]
for text, x in steps:
    ax.text(x, 0.5, text, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.7", fill=False, linewidth=1.4))
for i in range(len(steps) - 1):
    ax.annotate("", xy=(steps[i + 1][1] - 0.065, 0.5), xytext=(steps[i][1] + 0.065, 0.5),
                arrowprops=dict(arrowstyle="->", linewidth=1.4))
ax.set_title("Temporal Fraud Detection — implemented pipeline")
fig.tight_layout()
fig.savefig(OUT / "pipeline_overview.png", dpi=180, bbox_inches="tight")
plt.close(fig)

# 2) Causal graph illustration
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.axis("off")
points = {"T1": (0.08, 0.55), "T2": (0.08, 0.25), "T3": (0.38, 0.55), "T4": (0.62, 0.35), "T5": (0.88, 0.55)}
edges = [("T1", "T3"), ("T2", "T3"), ("T3", "T4"), ("T1", "T4"), ("T2", "T5"), ("T4", "T5")]
for a, b in edges:
    x1, y1 = points[a]; x2, y2 = points[b]
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", linewidth=1.2))
for label, (x, y) in points.items():
    ax.text(x, y, label, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="circle,pad=0.35", fill=False, linewidth=1.3))
ax.text(0.5, 0.04, "Every edge points from an earlier transaction to a later transaction", ha="center", fontsize=10)
ax.set_title("Causal transaction graph")
fig.tight_layout()
fig.savefig(OUT / "temporal_graph_concept.png", dpi=180, bbox_inches="tight")
plt.close(fig)
