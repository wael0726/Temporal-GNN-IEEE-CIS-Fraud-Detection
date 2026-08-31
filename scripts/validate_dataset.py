from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED = {
    "train_transaction.csv": {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1", "addr1", "ProductCD"},
    "train_identity.csv": {"TransactionID", "DeviceInfo"},
    "test_transaction.csv": {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "ProductCD"},
    "test_identity.csv": {"TransactionID", "DeviceInfo"},
    "sample_submission.csv": {"TransactionID", "isFraud"},
}


def main(data_dir: str):
    root = Path(data_dir)
    for name, required in REQUIRED.items():
        path = root / name
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        cols = set(pd.read_csv(path, nrows=0).columns)
        missing = sorted(required - cols)
        if missing:
            raise SystemExit(f"{name}: missing columns: {missing}")
        print(f"OK  {name:24s} {path.stat().st_size / 1024 / 1024:8.1f} MiB")
    print("Dataset structure validated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    main(args.data_dir)
