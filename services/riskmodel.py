# services/riskmodel.py
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _safe_minmax(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return arr
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def compute_rule_violation_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based expert system layer.
    Output: rule_violation_score ∈ [0,1] + rule_reasons list
    """
    out = df.copy()

    # Ensure required columns exist
    for col in ["avg_kwh", "std_kwh", "zero_ratio", "drop_ratio", "spike_ratio"]:
        if col not in out.columns:
            out[col] = 0.0

    reasons = []
    scores = []

    for _, row in out.iterrows():
        r = []
        s = 0.0

        # Sudden drop > 60%
        if row["drop_ratio"] >= 0.60:
            s += 0.35
            r.append("Sudden drop > 60% (possible bypass/tamper)")

        # Too many zero readings
        if row["zero_ratio"] >= 0.25:
            s += 0.25
            r.append("High zero readings (possible meter bypass/offline)")

        # Flat usage curve (low std)
        if row["avg_kwh"] > 0 and row["std_kwh"] / (row["avg_kwh"] + 1e-9) < 0.08:
            s += 0.20
            r.append("Flat usage curve (suspicious constant reading)")

        # Spike pattern
        if row["spike_ratio"] >= 0.50:
            s += 0.20
            r.append("High spikes detected (irregular load pattern)")

        # clamp
        s = min(1.0, s)
        reasons.append(r)
        scores.append(s)

    out["rule_violation_score"] = scores
    out["rule_reasons"] = reasons
    return out


def classical_ml_scores(features: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Classical ML layer:
    - Isolation Forest
    - LOF
    Output: classical_ml_score ∈ [0,1]
    """
    X = features.copy()

    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Isolation Forest
    iforest = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=random_state,
    )
    iforest.fit(X)
    # decision_function: higher = more normal
    if_scores = -iforest.decision_function(X)  # higher = more anomalous
    if_norm = _safe_minmax(if_scores)

    # LOF (novelty=False for fit_predict on same data)
    lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(X) - 1)), contamination=0.08)
    lof_labels = lof.fit_predict(X)
    lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous
    lof_norm = _safe_minmax(lof_scores)

    classical = 0.5 * if_norm + 0.5 * lof_norm

    return pd.DataFrame(
        {
            "iforest_score": if_norm,
            "lof_score": lof_norm,
            "classical_ml_score": classical,
        }
    )


def meta_ensemble_decision(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision Layer:
    - Combine Rules + Classical ML
    - Penalize disagreement (uncertainty)
    Output:
    - final_risk ∈ [0,1]
    - risk_score 0-100
    - label: Normal / Monitor / Inspection
    - confidence ∈ [0,1]
    """
    out = df.copy()

    for col in ["classical_ml_score", "rule_violation_score"]:
        if col not in out.columns:
            out[col] = 0.0

    # disagreement between ML and rules
    disagreement = np.abs(out["classical_ml_score"] - out["rule_violation_score"])
    uncertainty_penalty = _safe_minmax(disagreement)
    out["uncertainty_penalty"] = uncertainty_penalty

    # base risk (without penalty)
    base_risk = (0.60 * out["classical_ml_score"] + 0.40 * out["rule_violation_score"]).clip(0, 1)

    # final risk = base risk - penalty
    out["final_risk"] = (base_risk - 0.20 * uncertainty_penalty).clip(0, 1)

    # risk score (0-100)
    out["risk_score"] = (out["final_risk"] * 100).round(2)

    # label policy (0-100 scale)
    labels = []
    actions = []

    for score in out["risk_score"]:
        if score < 40:
            labels.append("Normal")
            actions.append("No Action")
        elif score < 70:
            labels.append("Monitor")
            actions.append("Monitor / Verify")
        else:
            labels.append("Inspection")
            actions.append("Inspection Priority")

    out["label"] = labels
    out["action"] = actions

    # confidence:
    # high confidence when both agree + risk is extreme (very low or very high)
    agreement = 1 - disagreement
    extremeness = np.maximum(out["final_risk"], 1 - out["final_risk"])  # close to 0 or 1 => high
    confidence = 0.65 * agreement + 0.35 * extremeness
    out["confidence"] = confidence.clip(0, 1).round(3)

    return out


def run_hybrid_risk_engine(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline:
    1) Rule layer
    2) Classical ML layer (IF + LOF)
    3) Meta ensemble
    """
    df = features_df.copy()

    # Rule layer
    df = compute_rule_violation_score(df)

    # Classical ML layer
    ml_df = classical_ml_scores(df[["avg_kwh", "std_kwh", "zero_ratio", "drop_ratio", "spike_ratio"]])
    df = pd.concat([df.reset_index(drop=True), ml_df.reset_index(drop=True)], axis=1)

    # Decision layer
    df = meta_ensemble_decision(df)

    return df
