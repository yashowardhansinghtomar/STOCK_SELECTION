"""Generic data‑quality utilities.

Expected functions
------------------
check_missing(df, *, threshold=0.2)      -> dict|DataFrame
class_balance(df, label_col, *, max_skew=0.9) -> float
detect_outliers(df, cols=None, method='iqr')  -> DataFrame mask

These helpers must NOT call print() or logger; just return results.
"""

# core/validation/data_checks.py
import pandas as pd
import numpy as np
from scipy.stats import zscore


def check_missing(df, *, threshold=0.2):
    missing_ratio = df.isna().mean()
    return missing_ratio[missing_ratio > threshold].to_dict()


def class_balance(df, label_col, *, max_skew=0.9):
    counts = df[label_col].value_counts(normalize=True)
    return counts.max()


def detect_outliers(df, cols=None, method='iqr'):
    if cols is None:
        cols = df.select_dtypes(include='number').columns
    outliers = pd.DataFrame(False, index=df.index, columns=cols)

    for col in cols:
        if method == 'zscore':
            z = np.abs(zscore(df[col].dropna()))
            outliers[col] = False
            outliers.loc[df[col].dropna().index, col] = z > 3
        elif method == 'iqr':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers[col] = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))

    return outliers
