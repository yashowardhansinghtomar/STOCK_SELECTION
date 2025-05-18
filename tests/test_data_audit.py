import sys
import json
import pandas as pd
from pathlib import Path
from core import data_provider
from analysis.data_audit import main as audit_main

def test_data_audit_threshold_behavior(monkeypatch, tmp_path):
    def mock_load_data(name):
        if name == "training_data":
            return pd.DataFrame({
                "feature1": [1, 2, 3, 4],
                "feature2": [None, None, None, None],  # 100% missing
                "label": [1, 0, 1, 0]  # needed for class_balance
            }).astype({"feature2": "float64"})
        elif name == "stock_fundamentals":
            return pd.DataFrame()
        return pd.DataFrame()

    monkeypatch.setattr(data_provider, 'load_data', mock_load_data)

    # Inject CLI args for the audit CLI parser
    report_file = tmp_path / "report.json"
    sys.argv = [
        "data_audit",
        "--save-report", str(report_file),
        "--missing-threshold", "0.1"
    ]

    try:
        audit_main()
    except SystemExit as e:
        assert e.code == 1  # audit failed due to high missingness

    assert report_file.exists()

    report = json.loads(report_file.read_text())
    print("🔍 Report missing_train:", json.dumps(report["missing_train"], indent=2))
    assert "feature2" in report["missing_train"]
