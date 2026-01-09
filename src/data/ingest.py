import os
from typing import Union
import pandas as pd


def save_canonical(df: pd.DataFrame, out_dir: str, client_id: Union[str, int] = "client") -> str:
    """Save a pandas DataFrame into canonical format under out_dir.

    - df: DataFrame with time-series rows and sensor columns (timestamp optional).
    - out_dir: directory under which the file will be written (e.g., data/raw/)
    - client_id: used to name the output file: {client_id}.parquet

    Returns the path to the written file.
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{client_id}.parquet"
    path = os.path.join(out_dir, filename)
    # Prefer parquet if pandas supports it; fallback to csv
    try:
        df.to_parquet(path)
    except Exception:
        csv_path = os.path.join(out_dir, f"{client_id}.csv")
        df.to_csv(csv_path, index=False)
        path = csv_path
    return path
