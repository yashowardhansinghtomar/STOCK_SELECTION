"""
Shared Pytest fixtures.

Add lightweight fixtures here—e.g. DataFrame factories, temporary file
helpers, monkey‑patched config switches—so all test modules can import
them without duplication.
"""
import pytest
import pandas as pd

@pytest.fixture
def dummy_training_data():
    """Very small fixture DataFrame for unit tests."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "date": ["2024‑01‑01", "2024‑01‑02"],
            "label": [1, 0],
            "price": [100.0, 200.0],
            "as_of_date": ["2024‑01‑01", "2024‑01‑02"],
        }
    )
