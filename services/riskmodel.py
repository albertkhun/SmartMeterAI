import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def run_isolation_forest(features: pd.DataFrame) -> pd.DataFrame:
    X = features[["avg_kwh", "std_kwh", "max_kwh", "min_kwh", "drop_percent", "spike_percent"]].copy()

    model = IsolationForest(
        n_estimators=150,
        contamination=0.15,
        random_state=42
    )

    model.fit(X)

    preds = model.predict(X)  # 1 normal, -1 anomaly
    scores = model.decision_function(X)  # higher = normal, lower = anomaly

    features["prediction"] = preds
    features["anomaly_score"] = scores

    # Convert anomaly_score -> risk_score (0-100)
    # Lower score => higher risk
    scaled = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    features["risk_score"] = (100 - (scaled * 100)).round(2)

    # Labels
    features["label"] = features["prediction"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

    return features
