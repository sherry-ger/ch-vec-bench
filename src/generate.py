import random
from datetime import datetime, timedelta, timezone

import h3
import numpy as np

_LOCATIONS = ["oslo", "bergen", "trondheim", "stavanger", "tromso"]

# Norway bounding box
_LAT_MIN, _LAT_MAX = 57.0, 71.0
_LON_MIN, _LON_MAX = 4.0, 31.0


def generate_vectors(
    n: int,
    dims: int,
    dtype: str = "float32",
    n_queries: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate L2-normalized random vectors split into db and query sets.

    Returns (db_vectors, query_vectors) with shapes (n, dims) and (n_queries, dims).
    db and query are drawn from the same seeded pool with no overlap.
    """
    np_dtype = np.float32 if dtype == "float32" else np.float64
    rng = np.random.default_rng(seed)

    total = n + n_queries
    raw = rng.standard_normal((total, dims)).astype(np_dtype)

    # L2-normalize each row
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    normalized = raw / norms

    db_vectors = normalized[:n]
    query_vectors = normalized[n:]
    return db_vectors, query_vectors


def vectors_to_rows(
    vectors: np.ndarray,
    id_offset: int = 0,
    metadata: list[dict] | None = None,
) -> list[dict]:
    """Convert a (N, dims) array to a list of row dicts for ClickHouse insert.

    Each dict has keys: id, ts, lat, lon, h3_9, location, vec.
    vec is a plain Python list (not a numpy array).

    If metadata is None, synthetic Norway geo/time metadata is generated.
    If metadata is provided, it is merged; h3_9 is computed from lat/lon if absent.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)  # naive UTC for ClickHouse DateTime
    rows = []

    for i, vec in enumerate(vectors):
        if metadata is not None:
            m = dict(metadata[i]) if i < len(metadata) else {}
        else:
            m = {}

        if not m:
            lat = random.uniform(_LAT_MIN, _LAT_MAX)
            lon = random.uniform(_LON_MIN, _LON_MAX)
            m = {
                "ts": now - timedelta(seconds=random.randint(0, 30 * 24 * 3600)),
                "lat": lat,
                "lon": lon,
                "h3_9": int(h3.latlng_to_cell(lat, lon, 9), 16),
                "location": random.choice(_LOCATIONS),
            }
        elif "lat" in m and "lon" in m and "h3_9" not in m:
            m["h3_9"] = int(h3.latlng_to_cell(m["lat"], m["lon"], 9), 16)

        rows.append({
            "id": id_offset + i,
            "ts": m.get("ts", now),
            "lat": m.get("lat", 0.0),
            "lon": m.get("lon", 0.0),
            "h3_9": m.get("h3_9", 0),
            "location": m.get("location", ""),
            "vec": vec.tolist(),
        })

    return rows


def load_vectors_file(
    path: str,
    vector_col: str = "",
    normalize: bool = True,
) -> np.ndarray:
    """Load vectors from a .npy, .csv, or .parquet file.

    Returns shape (N, dims) float64, optionally L2-normalized.

    For parquet: auto-detects a column named 'vec', 'vector', 'embedding', or
    'embeddings'. If the column contains lists/arrays they are stacked; otherwise
    all numeric columns are used. Override with vector_col.
    For csv: assumes no header row, one vector per line.
    """
    import pandas as pd
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")

    ext = p.suffix.lower()

    if ext == ".npy":
        vectors = np.load(path).astype(np.float64)

    elif ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        # Try without header first; if first value isn't a float, assume header row
        try:
            first = open(path).readline().split(sep)[0].strip()
            float(first)
            header = None
        except ValueError:
            header = 0
        vectors = pd.read_csv(path, header=header, sep=sep).values.astype(np.float64)

    elif ext == ".parquet":
        df = pd.read_parquet(path)
        col = vector_col
        if not col:
            for candidate in ("vec", "vector", "embedding", "embeddings"):
                if candidate in df.columns:
                    col = candidate
                    break
        if col and col in df.columns:
            sample = df[col].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                vectors = np.stack(df[col].tolist()).astype(np.float64)
            else:
                vectors = df[[col]].values.astype(np.float64)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(
                    f"No numeric columns found in {path}. "
                    "Use --vector-col to specify the column containing vectors."
                )
            vectors = df[numeric_cols].values.astype(np.float64)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npy, .csv, or .parquet")

    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {vectors.shape}")

    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        vectors = vectors / norms

    print(f"Loaded {vectors.shape[0]:,} × {vectors.shape[1]}-dim vectors from {p.name} (dtype=float64)")
    return vectors


if __name__ == "__main__":
    db, query = generate_vectors(1000, 128)
    print(f"db shape:    {db.shape}  dtype: {db.dtype}")
    print(f"query shape: {query.shape}")
    print(f"norm[0]:     {np.linalg.norm(db[0]):.6f}  (expect 1.0)")

    rows = vectors_to_rows(db)
    sample = {k: v if k != "vec" else f"[...{len(v)} floats...]" for k, v in rows[0].items()}
    print(f"sample row:  {sample}")
    print(f"ts type:     {type(rows[0]['ts']).__name__}")
    print(f"h3_9 type:   {type(rows[0]['h3_9']).__name__}")
    print(f"vec type:    {type(rows[0]['vec']).__name__}")
