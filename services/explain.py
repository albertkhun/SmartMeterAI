# services/explain.py
def build_reason_text(row) -> str:
    """
    Human-readable explainability for PPT + dashboard.
    """
    reasons = []

    # Add rule reasons if present
    if "rule_reasons" in row and isinstance(row["rule_reasons"], list):
        reasons.extend(row["rule_reasons"])

    # Add model insight
    if "classical_ml_score" in row:
        if row["classical_ml_score"] >= 0.75:
            reasons.append("Classical ML models detected strong anomaly pattern")
        elif row["classical_ml_score"] >= 0.45:
            reasons.append("ML detected moderate anomaly pattern")

    # Confidence messaging
    if "confidence" in row:
        if row["confidence"] < 0.45:
            reasons.append("Low confidence: needs more data / manual verification")

    if not reasons:
        return "Consumption pattern looks normal based on current data."

    # Keep it short for UI
    return "; ".join(reasons[:4])
