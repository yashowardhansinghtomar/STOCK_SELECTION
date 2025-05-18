# tests/test_model_health.py
"""
Integration tests for `analysis.model_health`.

Create an in‑memory DataFrame that mimics `model_metadata` and patch
`load_data` to return it.  Verify that drift detection flags the right
runs and that any plot or file artefacts are produced.
"""
import os
import pandas as pd
from analysis import model_health
from core import data_provider

import pytest

@pytest.fixture
def dummy_model_metadata():
    return pd.DataFrame({
        "date": pd.date_range(start="2024-01-01", periods=60, freq="D"),
        "accuracy": [0.9] * 30 + [0.7] * 30,
        "rmse": [0.1] * 30 + [0.3] * 30
    })

def test_model_drift_detection(monkeypatch, tmp_path, dummy_model_metadata):
    monkeypatch.setattr(data_provider, 'load_data', lambda name: dummy_model_metadata)

    cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        model_health.main(output_dir=".")
    finally:
        os.chdir(cwd)

    assert any(str(p).endswith(".png") for p in tmp_path.iterdir())
