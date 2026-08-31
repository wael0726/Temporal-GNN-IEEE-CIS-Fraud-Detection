from __future__ import annotations

from collections import defaultdict, deque
import numpy as np
import pandas as pd

ENTITY_COLUMNS = ["card1", "addr1", "DeviceInfo", "P_emaildomain", "ProductCD"]
MISSING = "<MISSING>"


def _clean_entity_values(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    raw = series.astype("string")
    valid = raw.notna().to_numpy()
    values = raw.fillna(MISSING).astype(str).to_numpy()
    return values, valid


def _past_count(series: pd.Series) -> pd.Series:
    """Count prior *known* occurrences; missing values never form an entity."""
    values, valid = _clean_entity_values(series)
    counts: dict[str, int] = {}
    out = np.zeros(len(series), dtype=np.float32)
    for i, (key, is_valid) in enumerate(zip(values, valid)):
        if is_valid:
            out[i] = counts.get(key, 0)
            counts[key] = counts.get(key, 0) + 1
    return pd.Series(out, index=series.index)


def _past_count_window(series: pd.Series, timestamps: pd.Series, seconds: int) -> pd.Series:
    """Causal rolling count of prior known events in a time window."""
    values, valid = _clean_entity_values(series)
    ts = pd.to_datetime(timestamps).astype("int64").to_numpy() // 10**9
    queues: dict[str, deque[int]] = defaultdict(deque)
    out = np.zeros(len(series), dtype=np.float32)
    for i, (key, t, is_valid) in enumerate(zip(values, ts, valid)):
        if not is_valid:
            continue
        q = queues[key]
        cutoff = int(t) - seconds
        while q and q[0] < cutoff:
            q.popleft()
        out[i] = len(q)
        q.append(int(t))
    return pd.Series(out, index=series.index)


def build_temporal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create strictly causal transaction features.

    Every historical count excludes the current event and missing categorical
    values are treated as unknown, not as a shared entity.
    """
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError("Temporal features require chronologically sorted input.")

    # TransactionDT is an elapsed-time field; the exact calendar anchor is not
    # known, so we use time-of-day and a relative weekly cycle instead of claiming
    # that the synthetic date corresponds to a real weekday.
    seconds = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype(float)
    relative_day = ((ts - ts.iloc[0]).dt.total_seconds() / 86400.0).astype(float) if "TransactionDT" not in out.columns else (pd.to_numeric(out["TransactionDT"], errors="coerce") / 86400.0).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * seconds / 86400).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * seconds / 86400).astype("float32")
    out["week_cycle_sin"] = np.sin(2 * np.pi * relative_day / 7).astype("float32")
    out["week_cycle_cos"] = np.cos(2 * np.pi * relative_day / 7).astype("float32")
    out["log_amount"] = np.log1p(pd.to_numeric(out["amount"], errors="coerce").clip(lower=0)).astype("float32")

    features = ["log_amount", "hour_sin", "hour_cos", "week_cycle_sin", "week_cycle_cos"]

    for col in ENTITY_COLUMNS:
        if col in out.columns:
            out[f"{col}_past_count"] = _past_count(out[col]).astype("float32")
            out[f"{col}_count_1h"] = _past_count_window(out[col], ts, 3600).astype("float32")
            features += [f"{col}_past_count", f"{col}_count_1h"]

    if "card1" in out.columns:
        card = out["card1"].astype("string")
        valid = card.notna()
        prev_ts = ts.where(valid).groupby(card, sort=False).shift(1)
        out["card_seconds_since_prev"] = (
            ts - prev_ts
        ).dt.total_seconds().where(valid, 0).fillna(1e9).clip(lower=0).astype("float32")
        out["card_count_24h"] = _past_count_window(card, ts, 86400).astype("float32")
        features += ["card_seconds_since_prev", "card_count_24h"]

    out.loc[:, features] = out[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, features
