import pandas as pd

def clamp01(x: float) -> float:
    try:
        x = float(x)
    except:
        x = 0.0
    return max(0.0, min(1.0, x))


def compute_layer_scores(row: pd.Series) -> dict:
    """
    Returns 6 layer scores in [0,1] per consumer.
    Prototype-safe (works even if some feature columns are missing).
    """

    # L1 Physics / Energy Balance (proxy)
    l1 = row.get("energy_loss_score", row.get("anomaly_score", 0.0))
    l1 = clamp01(l1)

    # L2 Rule-based Legal Layer (proxy)
    rule_score = row.get("rule_violation_score", None)
    if rule_score is None:
        sudden_drop = row.get("sudden_drop_flag", 0)
        zeros = row.get("zero_ratio", 0.0)
        rule_score = 0.6 * float(sudden_drop) + 0.4 * float(zeros)
    l2 = clamp01(rule_score)

    # L3 Graph / Collusion (future-ready placeholder)
    neighbor_dev = row.get("neighbor_deviation", row.get("std_kwh", 0.0))
    neighbor_dev = float(neighbor_dev) if neighbor_dev is not None else 0.0
    l3 = clamp01(neighbor_dev / (neighbor_dev + 10.0))

    # L4 ML layer (IsolationForest + LOF ensemble)
    l4 = row.get("ml_anomaly_score", row.get("anomaly_score", 0.0))
    l4 = clamp01(l4)

    # L5 Evidence Layer (future: drone, seal tamper, image)
    l5 = clamp01(row.get("evidence_score", 0.0))

    # L6 Human Review Layer
    human_status = str(row.get("human_status", "pending")).lower()
    if human_status in ["confirmed", "confirmed theft", "theft"]:
        l6 = 1.0
    elif human_status in ["cleared", "false alarm", "normal"]:
        l6 = 0.0
    else:
        l6 = 0.5

    return {
        "layer1_physics": l1,
        "layer2_rules": l2,
        "layer3_collusion": l3,
        "layer4_ml": l4,
        "layer5_evidence": l5,
        "layer6_human": l6,
    }


def compute_final_risk(layer_scores: dict) -> float:
    """
    Weighted final risk score (0-100).
    """
    weights = {
        "layer1_physics": 0.25,
        "layer2_rules": 0.20,
        "layer3_collusion": 0.15,
        "layer4_ml": 0.30,
        "layer5_evidence": 0.05,
        "layer6_human": 0.05,
    }

    score01 = 0.0
    for k, w in weights.items():
        score01 += w * float(layer_scores.get(k, 0.0))

    return round(100.0 * clamp01(score01), 2)
