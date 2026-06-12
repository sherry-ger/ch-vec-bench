import asyncio
import time
from dataclasses import dataclass

import clickhouse_connect
import numpy as np

from src.config import (
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE,
)

DB = CLICKHOUSE_DATABASE

_QBIT_SETTINGS = {"allow_experimental_qbit_type": 1}
_MAX_BITS = {"float32": 32, "float64": 64}


@dataclass
class BenchmarkResult:
    strategy: str
    dims: int
    n_vectors: int
    top_k: int
    median_latency_ms: float
    p95_latency_ms: float
    recall_at_k: float           # 0.0–1.0, mean over all queries
    compressed_bytes: int        # from system.columns
    uncompressed_bytes: int      # from system.columns
    compression_ratio: float     # uncompressed / compressed


def _recall(retrieved: list[int], truth: list[int]) -> float:
    return len(set(retrieved) & set(truth)) / len(truth) if truth else 0.0


async def measure_storage(client, table: str) -> tuple[int, int]:
    """Returns (compressed_bytes, uncompressed_bytes) from system.parts.

    Uses system.parts (not system.columns) because on ClickHouse Cloud
    SharedMergeTree, system.columns.data_compressed_bytes is always 0.
    """
    r = await client.query(
        "SELECT sum(data_compressed_bytes), sum(data_uncompressed_bytes) "
        "FROM system.parts "
        "WHERE table = {t:String} AND database = {db:String} AND active = 1",
        parameters={"t": table, "db": DB},
    )
    row = r.result_rows[0]
    return int(row[0] or 0), int(row[1] or 0)


async def get_n_vectors(client, table: str) -> int:
    r = await client.query(f"SELECT count() FROM {DB}.{table}")
    return int(r.result_rows[0][0])


async def get_ground_truth(
    client,
    query_vectors: np.ndarray,
    top_k: int,
    where_clause: str = "",
) -> list[list[int]]:
    """Exact KNN on vec_brute_f32 — one top-K id list per query vector."""
    results = []
    for vec in query_vectors:
        sql = (
            f"SELECT id FROM {DB}.vec_brute_f32 {where_clause} "
            f"ORDER BY cosineDistance(vec, {{query:Array(Float32)}}) "
            f"LIMIT {{k:UInt32}}"
        )
        r = await client.query(sql, parameters={"query": vec.tolist(), "k": top_k})
        results.append([row[0] for row in r.result_rows])
    return results


async def _run_queries(
    client,
    sql_template: str,
    query_vectors: np.ndarray,
    ground_truth: list[list[int]],
    extra_params: dict | None = None,
    settings: dict | None = None,
) -> tuple[list[float], float]:
    """Run sql_template for each query vector. Returns (latencies_ms, mean_recall)."""
    latencies: list[float] = []
    recalls: list[float] = []

    for vec, truth in zip(query_vectors, ground_truth):
        params = {"query": vec.tolist(), **(extra_params or {})}
        t0 = time.perf_counter()
        r = await client.query(sql_template, parameters=params, settings=settings)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        retrieved = [row[0] for row in r.result_rows]
        recalls.append(_recall(retrieved, truth))

    return latencies, float(np.mean(recalls))


def _make_result(
    strategy: str,
    dims: int,
    n_vectors: int,
    top_k: int,
    latencies: list[float],
    mean_recall: float,
    compressed: int,
    uncompressed: int,
) -> BenchmarkResult:
    ratio = uncompressed / compressed if compressed > 0 else 0.0
    return BenchmarkResult(
        strategy=strategy,
        dims=dims,
        n_vectors=n_vectors,
        top_k=top_k,
        median_latency_ms=float(np.median(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        recall_at_k=round(mean_recall, 4),
        compressed_bytes=compressed,
        uncompressed_bytes=uncompressed,
        compression_ratio=round(ratio, 3),
    )


async def bench_brute(
    client,
    table: str,
    query_vectors: np.ndarray,
    ground_truth: list[list[int]],
    top_k: int,
    where_clause: str = "",
) -> BenchmarkResult:
    arr_type = "Array(Float64)" if table.endswith("f64") else "Array(Float32)"
    # For f64 tables, cast query vectors
    qvecs = query_vectors.astype(np.float64) if table.endswith("f64") else query_vectors
    sql = (
        f"SELECT id FROM {DB}.{table} {where_clause} "
        f"ORDER BY cosineDistance(vec, {{query:{arr_type}}}) "
        f"LIMIT {{k:UInt32}}"
    )
    latencies, recall = await _run_queries(
        client, sql, qvecs, ground_truth,
        extra_params={"k": top_k},
    )
    dims = query_vectors.shape[1]
    n = await get_n_vectors(client, table)
    compressed, uncompressed = await measure_storage(client, table)
    return _make_result(table, dims, n, top_k, latencies, recall, compressed, uncompressed)


async def bench_qbit(
    client,
    query_vectors: np.ndarray,
    ground_truth: list[list[int]],
    top_k: int,
    bit_depths: list[int],
    dtype: str = "float32",
    where_clause: str = "",
) -> list[BenchmarkResult]:
    table = f"vec_qbit_{dtype.replace('float', 'f')}"
    max_bits = _MAX_BITS[dtype]
    arr_type = f"Array(Float32)" if dtype == "float32" else "Array(Float64)"
    qvecs = query_vectors.astype(np.float64) if dtype == "float64" else query_vectors

    dims = query_vectors.shape[1]
    n = await get_n_vectors(client, table)
    compressed, uncompressed = await measure_storage(client, table)

    results = []
    for k_bits in bit_depths:
        if k_bits > max_bits:
            print(f"  ⚠ qbit {dtype}: clamping {k_bits} → {max_bits} bits")
            k_bits = max_bits

        sql = (
            f"SELECT id FROM {DB}.{table} {where_clause} "
            f"ORDER BY cosineDistanceTransposed(vec, {{query:{arr_type}}}, {{bits:UInt8}}) "
            f"LIMIT {{k:UInt32}}"
        )
        latencies, recall = await _run_queries(
            client, sql, qvecs, ground_truth,
            extra_params={"k": top_k, "bits": k_bits},
            settings=_QBIT_SETTINGS,
        )
        strategy = f"qbit_{dtype.replace('float', 'f')}_{k_bits}b"
        results.append(_make_result(
            strategy, dims, n, top_k, latencies, recall, compressed, uncompressed
        ))
    return results


async def bench_vsim(
    client,
    table: str,
    query_vectors: np.ndarray,
    ground_truth: list[list[int]],
    top_k: int,
    where_clause: str = "",
) -> BenchmarkResult:
    # HNSW index is used automatically by the optimizer for cosineDistance queries
    sql = (
        f"SELECT id FROM {DB}.{table} {where_clause} "
        f"ORDER BY cosineDistance(vec, {{query:Array(Float32)}}) "
        f"LIMIT {{k:UInt32}}"
    )
    latencies, recall = await _run_queries(
        client, sql, query_vectors, ground_truth,
        extra_params={"k": top_k},
    )
    dims = query_vectors.shape[1]
    n = await get_n_vectors(client, table)
    compressed, uncompressed = await measure_storage(client, table)
    return _make_result(table, dims, n, top_k, latencies, recall, compressed, uncompressed)


async def run_all(
    client,
    query_vectors: np.ndarray,
    ground_truth: list[list[int]],
    dims: int,
    top_k: int,
    bit_depths: list[int] | None = None,
    dtype: str = "float32",
    where_clause: str = "",
) -> list[BenchmarkResult]:
    if bit_depths is None:
        bit_depths = [4, 8, 16, 32]

    results: list[BenchmarkResult] = []

    for table in ["vec_brute_f32", "vec_brute_f64"]:
        print(f"  benchmarking {table} ...")
        results.append(await bench_brute(client, table, query_vectors, ground_truth, top_k, where_clause))

    # Always benchmark both qbit tables — cast query vectors to float64 for f64 table
    for qbit_dtype in ["float32", "float64"]:
        print(f"  benchmarking vec_qbit_{qbit_dtype.replace('float', 'f')} ({bit_depths} bits) ...")
        results.extend(await bench_qbit(client, query_vectors, ground_truth, top_k, bit_depths, qbit_dtype, where_clause))

    for table in ["vec_vsim_f32", "vec_vsim_f16", "vec_vsim_i8"]:
        print(f"  benchmarking {table} ...")
        results.append(await bench_vsim(client, table, query_vectors, ground_truth, top_k, where_clause))

    return results


async def _main() -> None:
    from src.schema import drop_all_tables, ensure_tables
    from src.generate import generate_vectors, vectors_to_rows
    from src.loader import load_all

    dims, n_db, top_k = 128, 1000, 10

    client = await clickhouse_connect.get_async_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD,
        database=DB, secure=True,
    )
    try:
        print("Resetting tables ...")
        await drop_all_tables(client)
        await ensure_tables(client, dims=dims)

        print(f"\nLoading {n_db} vectors (dim={dims}) ...")
        db_vecs, query_vecs = generate_vectors(n_db, dims)
        rows = vectors_to_rows(db_vecs)
        await load_all(client, db_vecs, rows)

        print(f"\nComputing ground truth ({len(query_vecs)} queries) ...")
        ground_truth = await get_ground_truth(client, query_vecs, top_k)

        print("\nRunning benchmarks ...")
        results = await run_all(client, query_vecs, ground_truth, dims=dims, top_k=top_k)

        print("\n=== Results ===")
        for r in results:
            print(
                f"  {r.strategy:30s}"
                f"  recall={r.recall_at_k:.3f}"
                f"  p50={r.median_latency_ms:6.1f}ms"
                f"  p95={r.p95_latency_ms:6.1f}ms"
                f"  comp_ratio={r.compression_ratio:.2f}"
            )
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
