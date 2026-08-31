import pandas as pd
from src.features import build_temporal_features


def test_temporal_features_are_causal():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:01:00"]),
        "amount": [10.0, 20.0],
        "card1": ["A", "A"],
        "addr1": ["X", "X"],
        "DeviceInfo": ["D", "D"],
        "P_emaildomain": ["e", "e"],
        "ProductCD": ["W", "W"],
    })
    out, features = build_temporal_features(df)
    assert out.loc[0, "card1_past_count"] == 0
    assert out.loc[1, "card1_past_count"] == 1
    assert "log_amount" in features
