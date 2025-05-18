# tests/test_feature_pipeline.py
"""
Integration tests for `analysis.feature_pipeline`.

Focus on:
• Temporal alignment guard (future leakage).
• High‑correlation feature pruning.

Use two tiny DataFrames where leakage / correlation is easy to see.
"""
import numpy as np
import pandas as pd
import pytest
from analysis.feature_pipeline import check_temporal_alignment, detect_high_correlation

def test_temporal_alignment_pass():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'as_of_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
    })
    check_temporal_alignment(df)  # Should not raise

def test_temporal_alignment_fail():
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-03']),
        'as_of_date': pd.to_datetime(['2024-01-02', '2024-01-02'])
    })
    with pytest.raises(AssertionError):
        check_temporal_alignment(df)

def test_high_correlation_detection():
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [2, 4, 6, 8],   # perfectly correlated with 'a'
        'c': [10, 20, 10, 20]  # not correlated
    })
    correlated = detect_high_correlation(df, threshold=0.95)
    assert 'b' in correlated
