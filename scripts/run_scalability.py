from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import run


def main():
    p = argparse.ArgumentParser(description="Measure model quality and resource scaling across dataset sizes.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset", default="ieee_cis", choices=["ieee_cis", "ulb"])
    p.add_argument("--sizes", type=int, nargs="+", default=[10000, 50000, 100000])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--history-k", type=int, default=3)
    p.add_argument("--graph-mode", choices=["auto", "full", "sampled"], default="auto")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--output-dir", default="outputs/scalability")
    args = p.parse_args()

    process = psutil.Process(os.getpid())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for size in args.sizes:
        run_dir = out / f"{size//1000}k"
        rss_before = process.memory_info().rss
        t0 = time.perf_counter()
        result = run(
            data_dir=args.data_dir,
            dataset=args.dataset,
            max_rows=size,
            epochs=args.epochs,
            history_k=args.history_k,
            tune=args.tune,
            output_dir=run_dir,
            graph_mode=args.graph_mode,
        )
        elapsed = time.perf_counter() - t0
        rss_after = process.memory_info().rss
        rows.append({
            "rows": size,
            "elapsed_seconds": elapsed,
            "rss_before_mb": rss_before / 1024**2,
            "rss_after_mb": rss_after / 1024**2,
            "rss_delta_mb": (rss_after - rss_before) / 1024**2,
            "graph_edges": result["graph"]["edges"],
            "graph_mode": result["scalability"]["effective_graph_mode"],
            "xgb_test_pr_auc": result["xgboost"]["pr_auc"],
            "graphsage_test_pr_auc": result["graphsage"]["pr_auc"],
            "xgb_ap_lift": result["xgboost"]["ap_lift_vs_random"],
            "graphsage_ap_lift": result["graphsage"]["ap_lift_vs_random"],
        })

    (out / "scalability.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "scalability.csv", index=False)
    print(json.dumps({"runs": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
