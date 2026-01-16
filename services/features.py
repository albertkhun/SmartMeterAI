import numpy as np
import pandas as pd

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [c.strip().lower() for c in data.columns]

    # Find consumption column
    consumption_col = None
    for c in ["kwh", "units", "consumption", "energy"]:
        if c in data.columns:
            consumption_col = c
            break

    if consumption_col is None:
        raise ValueError("CSV must contain one column: kwh/units/consumption/energy")

    data[consumption_col] = pd.to_numeric(data[consumption_col], errors="coerce").fillna(0)
    data[consumption_col] = data[consumption_col].clip(lower=0)

    features = []
    for consumer_id, g in data.groupby("consumer_id"):
        vals = g[consumption_col].values.astype(float)

        avg = float(np.mean(vals))
        std = float(np.std(vals))
        zero_ratio = float(np.mean(vals == 0))

        n = len(vals)
        k = max(1, int(0.3 * n))
        first_avg = float(np.mean(vals[:k]))
        last_avg = float(np.mean(vals[-k:]))

        drop_ratio = 0.0
        if first_avg > 0:
            drop_ratio = max(0.0, (first_avg - last_avg) / (first_avg + 1e-9))

        spike_ratio = 0.0
        if avg > 0:
            spike_ratio = float(np.max(vals) / (avg + 1e-9))

        features.append({
            "consumer_id": consumer_id,
            "avg_kwh": round(avg, 3),
            "std_kwh": round(std, 3),
            "zero_ratio": round(zero_ratio, 3),
            "drop_ratio": round(drop_ratio, 3),
            "spike_ratio": round(spike_ratio, 3),
        })

    return pd.DataFrame(features)
