"""
Image descriptor loader from HuggingFace traffic-camera-norway-images.

Each image is resized to size×size grayscale pixels and flattened into a
float64 vector. No embedding model needed — pixel values ARE the descriptor.

Traffic labels are mapped to Norway cities for meaningful geo metadata.
"""
import random
from datetime import datetime, timedelta, timezone

import h3
import numpy as np
from PIL import Image

HF_DATASET = "ilsilfverskiold/traffic-camera-norway-images"

# Maps dataset label names → Norway city locations
LABEL_TO_LOCATION = {
    "high-traffic":   "oslo",
    "medium-traffic": "bergen",
    "low-traffic":    "trondheim",
    "no-traffic":     "tromso",
}

# lat_min, lat_max, lon_min, lon_max per city
LOCATION_BBOX = {
    "oslo":       (59.80, 59.97, 10.60, 10.90),
    "bergen":     (60.30, 60.50,  5.20,  5.40),
    "trondheim":  (63.30, 63.50, 10.30, 10.50),
    "tromso":     (69.60, 69.80, 18.90, 19.10),
}


def download_hf_images(n: int = 500, seed: int = 42) -> list[dict]:
    """Download n images from the HF dataset. Returns list of {image, label_name}."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")

    # Get label names from dataset features
    label_names = ds.features["label"].names  # ["high-traffic", ...]

    total = len(ds)
    if n >= total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), n))

    items = []
    for idx in indices:
        row = ds[idx]
        img = row["image"]
        label_name = label_names[row["label"]]
        items.append({"image": img, "label_name": label_name})

    return items


def image_to_descriptor(img: Image.Image, size: int = 8) -> np.ndarray:
    """Resize image to size×size grayscale, flatten to float64 array."""
    small = img.resize((size, size), Image.LANCZOS).convert("L")
    return np.array(small, dtype=np.float64).flatten()


def images_to_vectors(
    items: list[dict],
    size: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict]]:
    """Convert images to L2-normalized float64 descriptors + geo metadata.

    Returns (vectors shape (N, size*size), metadata list).
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    descriptors = []
    metadata = []

    for i, item in enumerate(items):
        desc = image_to_descriptor(item["image"], size)
        descriptors.append(desc)

        location = LABEL_TO_LOCATION.get(item["label_name"], "oslo")
        lat_min, lat_max, lon_min, lon_max = LOCATION_BBOX[location]
        lat = rng.uniform(lat_min, lat_max)
        lon = rng.uniform(lon_min, lon_max)
        h3_9 = int(h3.latlng_to_cell(lat, lon, 9), 16)
        ts = now - timedelta(seconds=rng.randint(0, 30 * 24 * 3600))

        metadata.append({
            "label_name": item["label_name"],
            "location": location,
            "lat": lat,
            "lon": lon,
            "h3_9": h3_9,
            "ts": ts,
        })

    vectors = np.stack(descriptors)  # shape (N, size*size), float64

    # L2-normalize each row for cosineDistance compatibility
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero for pure black images
    vectors = vectors / norms

    return vectors, metadata


def load_hf_descriptors(
    n: int = 500,
    size: int = 8,
    n_queries: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    """Download images, compute descriptors, split into db + query sets.

    Returns (db_vectors, query_vectors, db_metadata, query_metadata).
    """
    print(f"Downloading {n} images from {HF_DATASET} ...")
    items = download_hf_images(n, seed=seed)

    print(f"Computing {size}×{size} grayscale descriptors ({size * size}-dim float64) ...")
    vectors, metadata = images_to_vectors(items, size=size, seed=seed)

    n_db = n - n_queries
    db_vectors    = vectors[:n_db]
    query_vectors = vectors[n_db:]
    db_meta       = metadata[:n_db]
    query_meta    = metadata[n_db:]

    label_counts = {}
    for m in db_meta:
        label_counts[m["label_name"]] = label_counts.get(m["label_name"], 0) + 1

    print(
        f"  db:    {db_vectors.shape}  dtype={db_vectors.dtype}\n"
        f"  query: {query_vectors.shape}\n"
        f"  labels: {label_counts}"
    )
    return db_vectors, query_vectors, db_meta, query_meta


if __name__ == "__main__":
    db, query, db_meta, _ = load_hf_descriptors(n=20, size=8, n_queries=5)
    print(f"\nnorm[0] = {np.linalg.norm(db[0]):.6f}  (expect 1.0)")
    print(f"sample metadata: {db_meta[0]}")
