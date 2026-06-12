# ClickHouse Cloud Vector Benchmark

Benchmark every ClickHouse vector search strategy side-by-side: recall accuracy, query latency, and on-disk compression — using your own vectors or a built-in synthetic dataset.

```
┏━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Strategy      ┃ Dims ┃     N ┃ p50 ms ┃ p95 ms ┃ Recall ┃ Comp (MB) ┃ Raw (MB) ┃ Ratio ┃
┡━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ vec_brute_f64 │  128 │ 10000 │   50.0 │   59.1 │  1.000 │     0.668 │    1.069 │ 1.60x │
│ qbit_f64_4b   │  128 │ 10000 │   53.5 │   66.9 │  0.006 │     0.471 │    1.061 │ 2.25x │
│ vec_vsim_f16  │  128 │ 10000 │   49.0 │   58.9 │  0.979 │     0.502 │    0.557 │ 1.11x │
│ vec_vsim_i8   │  128 │ 10000 │   49.4 │   62.5 │  0.931 │     0.502 │    0.557 │ 1.11x │
└───────────────┴──────┴───────┴────────┴────────┴────────┴───────────┴──────────┴───────┘
```

## What it benchmarks

Seven strategies across three families:

| Strategy | Type | Description |
|---|---|---|
| `vec_brute_f32` | Exact KNN | `Array(Float32)` + `cosineDistance` full scan |
| `vec_brute_f64` | Exact KNN | `Array(Float64)` + `cosineDistance` full scan |
| `vec_qbit_f32_Kb` | Approx KNN | `QBit(Float32, dims)` queried at K bit planes |
| `vec_qbit_f64_Kb` | Approx KNN | `QBit(Float64, dims)` queried at K bit planes |
| `vec_vsim_f32` | ANN (HNSW) | `vector_similarity` index, no quantization |
| `vec_vsim_f16` | ANN (HNSW) | `vector_similarity` index, 16-bit quantization |
| `vec_vsim_i8` | ANN (HNSW) | `vector_similarity` index, 8-bit int quantization |

**Metrics:**
- **Recall@K** — what fraction of true nearest neighbours the strategy returns
- **p50/p95 latency** — query time in milliseconds (median and 95th percentile)
- **Compression ratio** — `uncompressed_bytes / compressed_bytes` from `system.parts`; higher = better on-disk compression

## Quick start

**Requirements:** Python 3.11+, a [ClickHouse Cloud](https://clickhouse.cloud) account (free tier works)

```bash
# 1. Clone
git clone https://github.com/your-org/clickhouse-vector-benchmark
cd clickhouse-vector-benchmark

# 2. Set up (creates .venv, installs deps, copies .env.example)
bash setup.sh

# 3. Fill in your ClickHouse Cloud credentials
#    Find them at: Settings → Connection details → HTTPS
nano .env          # or open in your editor

# 4. Run your first benchmark (synthetic vectors, no download needed)
source .venv/bin/activate
python -m src.main run --dims 128 --n-vectors 10000 --top-k 10
```

## Using your own vectors

If you have pre-computed embeddings, pass them directly:

```bash
# NumPy array (.npy) — shape (N, dims)
python -m src.main load --vectors-file my_embeddings.npy
python -m src.main bench --top-k 10 --export results.json

# CSV — no header, one vector per row
python -m src.main load --vectors-file my_embeddings.csv

# Parquet — auto-detects column named 'vec', 'vector', or 'embedding'
python -m src.main load --vectors-file my_embeddings.parquet

# Parquet with a custom column name
python -m src.main load --vectors-file my_embeddings.parquet --vector-col my_col
```

Dimensions and dtype are inferred automatically from the file. A random 20% of rows are held out as query vectors for recall measurement.

## All commands

```bash
python -m src.main --help

# One-shot: generate vectors + load + benchmark
python -m src.main run --dims 128 --n-vectors 10000 --dtype float32 --top-k 10

# Step by step
python -m src.main load --dims 128 --n-vectors 10000
python -m src.main bench --top-k 10 --qbit-bits 4,8,16,32 --export results.json

# Use your own vectors
python -m src.main load --vectors-file embeddings.parquet
python -m src.main bench --top-k 10

# Filter benchmarks to a subset of data
python -m src.main bench --top-k 10 --filter-location oslo
python -m src.main bench --top-k 10 --filter-after 2024-01-01
```

## Understanding the output

| Column | Meaning |
|---|---|
| **Recall** | Green ≥ 0.9 · Yellow 0.7–0.9 · Red < 0.7. Fraction of true top-K neighbours returned. 1.0 = exact. |
| **p50 / p95 ms** | Query latency at the 50th and 95th percentile across all query vectors. |
| **Comp (MB)** | On-disk compressed size (what you actually pay for in storage). |
| **Raw (MB)** | Uncompressed logical size. |
| **Ratio** | Raw / Compressed. Higher = ClickHouse compressed it more efficiently. |

> **QBit and synthetic vectors:** QBit recall is near-zero on random unit-normalized vectors because all components share the same floating-point exponent (no magnitude diversity). On real embeddings from models the recall improves significantly. Try `load-hf` for a real-data test.

## Try with a real image dataset

```bash
# Download and benchmark 500 Norway traffic camera images
# (no embedding model needed — uses 8×8 pixel descriptors)
python -m src.main load-hf --n 500
python -m src.main bench --top-k 10 --qbit-bits 4,8,12,16,32,64

# Filter to images tagged as high-traffic (mapped to Oslo)
python -m src.main bench --top-k 10 --filter-location oslo
```

## Export results

```bash
python -m src.main bench --top-k 10 --export results.json
python -m src.main bench --top-k 10 --export results.csv
```

Results include all `BenchmarkResult` fields: strategy, dims, n_vectors, top_k, latency stats, recall, storage sizes, and compression ratio.

## Docker

```bash
# Build
docker build -t vector-benchmark .

# Run with credentials from .env
docker run -it --env-file .env vector-benchmark run --dims 128 --n-vectors 5000

# Load your own vectors (mount a local directory)
docker run -it --env-file .env \
  -v $(pwd)/my_data:/data \
  vector-benchmark load --vectors-file /data/embeddings.parquet
```

## Architecture

```
src/
├── config.py     # ClickHouse connection from .env
├── schema.py     # CREATE TABLE DDL for all 7 strategies
├── generate.py   # Synthetic vector generation + file loading
├── loader.py     # Batched INSERT into ClickHouse
├── benchmark.py  # Latency / recall / storage measurement
├── report.py     # Rich table output + CSV/JSON export
├── main.py       # Typer CLI entry point
└── embed.py      # HuggingFace image descriptor loader
```

## Requirements

- Python 3.11+
- ClickHouse Cloud 26.x+ (required for QBit — `allow_experimental_qbit_type`)
- Older clusters work but QBit tables will be skipped gracefully
