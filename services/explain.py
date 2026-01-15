def generate_reason(row) -> str:
    reasons = []

    if row["drop_percent"] > 50:
        reasons.append("Sudden drop in usage (possible bypass/theft)")

    if row["spike_percent"] > 50:
        reasons.append("Sudden spike in usage (possible tampering/overload)")

    if row["std_kwh"] < 1 and row["avg_kwh"] > 0:
        reasons.append("Very flat usage pattern (meter stuck/tampered)")

    if row["avg_kwh"] < 1:
        reasons.append("Very low usage (suspicious zero/near-zero consumption)")

    if not reasons:
        reasons.append("Unusual pattern detected by anomaly model")

    return " | ".join(reasons)


def recommend_action(risk_score: float) -> str:
    if risk_score >= 80:
        return "Send inspection team now"
    elif risk_score >= 60:
        return "Schedule meter check"
    else:
        return "Monitor"
