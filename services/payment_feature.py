import pandas as pd
import numpy as np

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds payment mismatch features:
    - payment_delay_days
    - underpayment_ratio
    - fixed_payment_flag
    """

    out = df.copy()

    # convert to datetime
    if "due_date" in out.columns:
        out["due_date"] = safe_to_datetime(out["due_date"])
    if "paid_date" in out.columns:
        out["paid_date"] = safe_to_datetime(out["paid_date"])

    # payment delay
    if "due_date" in out.columns and "paid_date" in out.columns:
        out["payment_delay_days"] = (out["paid_date"] - out["due_date"]).dt.days
        out["payment_delay_days"] = out["payment_delay_days"].fillna(0).clip(lower=0)
    else:
        out["payment_delay_days"] = 0

    # underpayment ratio
    if "bill_amount" in out.columns and "paid_amount" in out.columns:
        out["underpayment_ratio"] = (out["bill_amount"] - out["paid_amount"]) / (out["bill_amount"] + 1e-6)
        out["underpayment_ratio"] = out["underpayment_ratio"].fillna(0).clip(lower=0)
    else:
        out["underpayment_ratio"] = 0

    # fixed payment flag (same amount repeatedly)
    if "paid_amount" in out.columns:
        out["fixed_payment_flag"] = out.groupby("consumer_id")["paid_amount"].transform(
            lambda x: 1 if x.nunique() <= 2 else 0
        )
    else:
        out["fixed_payment_flag"] = 0

    return out


def compute_cross_behavior_risk(df: pd.DataFrame, delay_threshold_days: int = 20) -> pd.DataFrame:
    """
    Returns consumer-level payment risk scores (0-1)
    """

    temp = df.copy()

    # time anomaly score
    temp["time_anomaly"] = (temp["payment_delay_days"] > delay_threshold_days).astype(int)

    # amount anomaly score
    temp["amount_anomaly"] = (temp["underpayment_ratio"] > 0.05).astype(int)

    # normalize delay and underpayment to 0-1
    delay_norm = temp["payment_delay_days"] / (temp["payment_delay_days"].max() + 1e-6)
    underpay_norm = temp["underpayment_ratio"].clip(0, 1)

    # usage anomaly (optional input)
    usage_anom = temp.get("usage_anomaly_score", pd.Series([0]*len(temp)))

    # final cross risk score (0-1)
    temp["cross_risk_score"] = (
        0.4 * usage_anom +
        0.3 * delay_norm +
        0.3 * underpay_norm
    ).clip(0, 1)

    # consumer-level aggregation
    agg = temp.groupby("consumer_id").agg({
        "cross_risk_score": "mean",
        "payment_delay_days": "mean",
        "underpayment_ratio": "mean",
        "fixed_payment_flag": "max",
        "time_anomaly": "max",
        "amount_anomaly": "max",
    }).reset_index()

    return agg
