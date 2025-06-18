from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from core.data_provider.data_provider import load_data
from core.logger.logger import logger

def check_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[features], current_data=current[features])
    result = report.as_dict()

    drift_detected = result['metrics'][0]['result']['dataset_drift']
    if drift_detected:
        logger.warning("🚨 Data drift detected! Immediate retraining recommended.")
        return True
    logger.info("✅ No significant data drift detected.")
    return False
