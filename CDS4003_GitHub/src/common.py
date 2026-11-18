"""Portable paths, configuration and serialization helpers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def config(path=None):
    return json.loads(Path(path or ROOT / "config.json").read_text())

def json_default(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (Path, pd.Timestamp)):
        return str(obj)
    raise TypeError(type(obj).__name__)

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default, allow_nan=False) + "\n")

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_split(path, limit=None, seed=42, usecols=None):
    """Uniform bottom-k hash sample, never the first k chronological rows."""
    chunks = []
    selected = None
    for chunk in pd.read_csv(path, chunksize=100000, low_memory=False, usecols=usecols):
        if limit is None:
            chunks.append(chunk)
        else:
            chunk["_priority"] = pd.util.hash_pandas_object(
                chunk["record_id"].astype(str) + str(seed), index=False
            ).to_numpy()
            selected = chunk if selected is None else pd.concat([selected, chunk], ignore_index=True)
            selected = selected.nsmallest(limit, "_priority")
    if limit is None:
        if not chunks:
            raise ValueError(f"No rows in {path}")
        selected = pd.concat(chunks, ignore_index=True)
    else:
        if selected is None or selected.empty:
            raise ValueError(f"No rows in {path}")
        selected = selected.drop(columns="_priority")
    return selected.sort_values(["flight_date", "record_id"]).reset_index(drop=True)
