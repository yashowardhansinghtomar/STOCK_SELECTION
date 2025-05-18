# tests/test_data_checks.py
"""
Unit tests for `core.validation.data_checks`.

Write *pure* unit tests for each helper:

• `check_missing()` — edge‑cases: no missing, all missing, some missing.
• `class_balance()` — binary vs. skewed labels; verify float output.
• `detect_outliers()` — compare z‑score vs. IQR masks.

Each test should assert on *return values* (don’t rely on prints/logs).
"""
import pandas as pd
import numpy as np
from core.validation.data_checks import check_missing, class_balance, detect_outliers

def test_check_missing():
    df = pd.DataFrame({
        'a': [1, None, None],
        'b': [1, 2, 3],
        'c': [None, None, None]
    })
    result = check_missing(df, threshold=0.5)
    assert 'a' in result
    assert 'c' in result
    assert 'b' not in result

def test_class_balance():
    df = pd.DataFrame({'label': [1, 1, 1, 0]})
    skew = class_balance(df, 'label', max_skew=0.9)
    assert abs(skew - 0.75) < 1e-6

def test_detect_outliers_zscore():
    df = pd.DataFrame({
        'x': [1, 1, 1, 100],
        'y': [10, 10, 10, 10]
    })
    outliers = detect_outliers(df, method='zscore')
    assert outliers['x'].iloc[-1] == True
    assert outliers['y'].any() == False

def test_detect_outliers_iqr():
    df = pd.DataFrame({
        'x': [1, 1, 1, 100],
        'y': [10, 10, 10, 10]
    })
    outliers = detect_outliers(df, method='iqr')
    assert outliers['x'].iloc[-1] == True
    assert outliers['y'].any() == False

def test_detect_outliers_zscore():
    df = pd.DataFrame({'x': [10]*20 + [10000], 'y': [5]*21})
    outliers = detect_outliers(df, method='zscore')
    # Convert np.bool_ to native bool for assertion
    assert bool(outliers['x'].iloc[-1]) is True


