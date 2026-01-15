import pandas as pd

REQUIRED_COLUMNS = ["consumer_id", "timestamp", "kwh"]

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop missing
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Convert kwh
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["kwh"])

    # Remove invalid kwh
    df = df[df["kwh"] >= 0]

    # Sort
    df = df.sort_values(by=["consumer_id", "timestamp"])

    return df
