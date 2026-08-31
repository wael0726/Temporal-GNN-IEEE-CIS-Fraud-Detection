from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

EXPECTED = {
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
}


def main(data_dir: str):
    root = Path(data_dir)
    archive = root / "ieee-fraud-detection.zip"
    if not archive.exists():
        raise SystemExit(f"Missing {archive}")
    with zipfile.ZipFile(archive) as z:
        names = {Path(n).name for n in z.namelist() if not n.endswith('/')}
        missing = EXPECTED - names
        if missing:
            raise SystemExit(f"Archive is missing: {sorted(missing)}")
        z.extractall(root)
    print("IEEE-CIS files extracted to:", root.resolve())
    for name in sorted(EXPECTED):
        print("  ", name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()
    main(args.data_dir)
