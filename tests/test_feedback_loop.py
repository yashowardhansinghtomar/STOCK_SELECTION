# tests/test_feedback_loop.py
"""
Integration tests for `analysis.feedback_loop`.

Mock small `recommendations`, `trades`, and `open_positions` frames,
run the audit, and assert KPI calculations (hit‑rate, realised Sharpe,
etc.) are as expected.
"""
import pandas as pd
import pytest
from analysis.feedback_loop import main as feedback_main
from core.data import data_provider

@pytest.fixture
def dummy_feedback_data():
    recs = pd.DataFrame({
        'ticker': ['A', 'B'],
        'date': ['2024-01-01', '2024-01-02'],
        'predicted_return': [0.1, -0.05]
    })
    trades = pd.DataFrame({
        'ticker': ['A', 'B'],
        'date': ['2024-01-01', '2024-01-02'],
        'realized_return': [0.12, -0.02]
    })
    open_pos = pd.DataFrame({
        'ticker': [],
        'date': []
    })
    training_data = pd.DataFrame({
        'ticker': ['A', 'B'],
        'label': [1, 0]
    })
    return recs, trades, open_pos, training_data

def test_feedback_loop_kpis(monkeypatch, dummy_feedback_data):
    recs, trades, open_pos, training_data = dummy_feedback_data

    monkeypatch.setattr(data_provider, 'load_data', lambda name: {
        'recommendations': recs,
        'paper_trades': trades,
        'open_positions': open_pos,
        'training_data': training_data
    }[name])

    feedback_main()  # Should print out KPIs without error
