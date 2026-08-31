import numpy as np
from src.evaluate import best_threshold, metrics


def test_threshold_is_selected_on_validation_and_reused_on_test():
    y_val = np.array([0, 0, 0, 1, 1, 1])
    p_val = np.array([0.05, 0.20, 0.45, 0.55, 0.80, 0.95])
    threshold = best_threshold(y_val, p_val)
    y_test = np.array([0, 1, 0, 1])
    p_test = np.array([0.40, 0.60, 0.90, 0.95])
    result = metrics(y_test, p_test, threshold=threshold)
    assert 0.01 <= threshold <= 0.99
    assert result["threshold"] == threshold


def test_explicit_threshold_is_not_reselected_on_test():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.6, 0.7, 0.8])
    assert metrics(y, p, threshold=0.5)["threshold"] == 0.5
