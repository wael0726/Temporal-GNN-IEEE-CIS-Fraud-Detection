from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

IEEE_TRANSACTION_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "isFraud",
]
IEEE_IDENTITY_COLUMNS = [
    "TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33",
    "id_36", "id_37", "id_38",
]


def _load_ieee_transactions(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = [c for c in IEEE_TRANSACTION_COLUMNS if c in header]
    required = {"TransactionID", "TransactionDT", "TransactionAmt", "isFraud"}
    missing = sorted(required - set(cols))
    if missing:
        raise ValueError(f"train_transaction.csv is missing required columns: {missing}")
    # Load all selected columns, then sort before applying max_rows. This makes
    # max_rows a true chronological prefix instead of assuming Kaggle file order.
    return pd.read_csv(path, usecols=cols)


def load_ieee_cis(data_dir: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load IEEE-CIS training data as a chronological event table.

    ``max_rows`` is applied *after* chronological sorting, so experiments are
    reproducible even if the source CSV is reordered.
    """
    data_dir = Path(data_dir)
    tx_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(
            f"Missing {tx_path}. Put train_transaction.csv and train_identity.csv in {data_dir}."
        )

    tx = _load_ieee_transactions(tx_path)

    if id_path.exists():
        id_header = pd.read_csv(id_path, nrows=0).columns.tolist()
        id_cols = [c for c in IEEE_IDENTITY_COLUMNS if c in id_header]
        ident = pd.read_csv(id_path, usecols=id_cols)
        tx = tx.merge(ident, on="TransactionID", how="left", sort=False, validate="one_to_one")
    else:
        for c in IEEE_IDENTITY_COLUMNS[1:]:
            tx[c] = np.nan

    tx = tx.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    if max_rows is not None:
        tx = tx.head(max_rows).copy()

    tx["timestamp"] = pd.Timestamp("2017-01-01") + pd.to_timedelta(tx["TransactionDT"], unit="s")
    tx["amount"] = tx["TransactionAmt"].astype("float32")
    tx["label"] = tx["isFraud"].astype("int8")
    tx["row_id"] = np.arange(len(tx), dtype=np.int64)
    return tx


def load_ulb(data_dir: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load the ULB/Worldline benchmark as an optional tabular fallback."""
    path = Path(data_dir) / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    df = pd.read_csv(path)
    df = df.sort_values(["Time"], kind="stable").reset_index(drop=True)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    df["timestamp"] = pd.Timestamp("2013-09-01") + pd.to_timedelta(df["Time"], unit="s")
    df["amount"] = df["Amount"].astype("float32")
    df["label"] = df["Class"].astype("int8")
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df
