from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Download IEEE-CIS Fraud Detection via the Kaggle CLI.")
    p.add_argument("--output-dir", default="data")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", "ieee-fraud-detection", "-p", str(out)],
        check=True,
    )
    import zipfile
    zips = list(out.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Kaggle did not produce a ZIP archive.")
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(out)
    print(f"Dataset extracted to {out.resolve()}")


if __name__ == "__main__":
    main()
