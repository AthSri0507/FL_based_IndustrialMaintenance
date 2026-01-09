from pathlib import Path
from typing import List, Optional

import pandas as pd

from .client import Client


def load_clients_from_dir(partitions_dir: str, max_clients: int = 3) -> List[Client]:
    """Load up to `max_clients` client partitions from a directory.

    Partitions are expected to be files named with a client id prefix or similar. We load
    files in sorted order and keep at most `max_clients` clients to enforce small experiments.
    """
    p = Path(partitions_dir)
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix in {".parquet", ".csv"}])
    clients = []
    for i, f in enumerate(files[:max_clients]):
        # read file into DataFrame and create Client
        if f.suffix == ".parquet":
            df = pd.read_parquet(f)
        else:
            df = pd.read_csv(f)
        clients.append(Client(client_id=i, data=df))
    return clients
