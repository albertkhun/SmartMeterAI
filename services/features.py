import pandas as pd
import numpy as np

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # Convert to daily usage
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby(["consumer_id", "date"])["kwh"].sum().reset_index()
    daily = daily.sort_values(["consumer_id", "date"])

    # Feature engineering per consumer
    features = daily.groupby("consumer_id")["kwh"].agg(
        avg_kwh="mean",
        std_kwh="std",
        max_kwh="max",
        min_kwh="min",
        total_days="count"
    ).reset_index()

    # Fill std NaN with 0 (if only 1 day)
    features["std_kwh"] = features["std_kwh"].fillna(0)

    # Sudden drop/spike detection (simple)
    daily["prev_kwh"] = daily.groupby("consumer_id")["kwh"].shift(1)
    daily["change"] = daily["kwh"] - daily["prev_kwh"]

    drop = daily[daily["change"] < 0].groupby("consumer_id")["change"].min().reset_index()
    spike = daily[daily["change"] > 0].groupby("consumer_id")["change"].max().reset_index()

    drop.rename(columns={"change": "max_drop"}, inplace=True)
    spike.rename(columns={"change": "max_spike"}, inplace=True)

    features = features.merge(drop, on="consumer_id", how="left")
    features = features.merge(spike, on="consumer_id", how="left")

    features["max_drop"] = features["max_drop"].fillna(0)
    features["max_spike"] = features["max_spike"].fillna(0)

    # % change estimate
    features["drop_percent"] = np.where(
        features["avg_kwh"] > 0,
        abs(features["max_drop"]) / features["avg_kwh"] * 100,
        0
    )
    features["spike_percent"] = np.where(
        features["avg_kwh"] > 0,
        abs(features["max_spike"]) / features["avg_kwh"] * 100,
        0
    )

    return features
