import pandas as pd


def load_ohlc_csv(path: str) -> pd.DataFrame:
    """
    Load OHLC (or close-only) data from a CSV file.

    Requirements for CSV:
    - Must contain a 'date' column that can be parsed to datetime
    - Must contain a 'close' column (other columns are optional)
    """
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column")

    # Parse date and set as index
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Normalize column names to lower case
    df = df.rename(columns=str.lower)

    if "close" not in df.columns:
        raise ValueError("CSV must contain a 'close' column")

    return df


__all__ = ["load_ohlc_csv"]

