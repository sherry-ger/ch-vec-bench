import asyncio
import json
from pathlib import Path
from typing import Optional

import clickhouse_connect
import h3
import numpy as np
import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from src.benchmark import BenchmarkResult, get_ground_truth, run_all
from src.config import (
    CLICKHOUSE_DATABASE, CLICKHOUSE_HOST, CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT, CLICKHOUSE_USER, validate_config,
)
from src.generate import generate_vectors, vectors_to_rows
from src.loader import LoadResult, load_all
from src.report import export_results, print_results_table
from src.schema import create_all_tables, drop_all_tables, ensure_tables

app = typer.Typer(help="Vector compression benchmark harness for ClickHouse Cloud.")
_console = Console()

QUERY_STATE_NPY  = "vectors_query.npy"
QUERY_STATE_JSON = "vectors_query_meta.json"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

async def _make_client():
    validate_config()
    return await clickhouse_connect.get_async_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=True,
    )


def _save_query_state(query_vecs: np.ndarray, dims: int, dtype: str) -> None:
    np.save(QUERY_STATE_NPY, query_vecs)
    with open(QUERY_STATE_JSON, "w") as f:
        json.dump({"dims": dims, "dtype": dtype}, f)


def _load_query_state() -> tuple[np.ndarray, int, str]:
    if not Path(QUERY_STATE_NPY).exists() or not Path(QUERY_STATE_JSON).exists():
        raise typer.BadParameter(
            "Query state not found. Run `load` first to generate and save vectors."
        )
    query_vecs = np.load(QUERY_STATE_NPY)
    meta = json.load(open(QUERY_STATE_JSON))
    return query_vecs, meta["dims"], meta["dtype"]


def _build_where_clause(
    filter_location: str | None,
    filter_after: str | None,
    filter_h3: str | None,
) -> str:
    clauses: list[str] = []

    if filter_location:
        clauses.append(f"location = '{filter_location}'")

    if filter_after:
        clauses.append(f"ts >= '{filter_after}'")

    if filter_h3:
        parts = filter_h3.split(",")
        if len(parts) != 4:
            raise typer.BadParameter("--filter-h3 must be 'lat,lon,resolution,k_ring'")
        lat, lon, res, k = float(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])
        center = h3.latlng_to_cell(lat, lon, res)
        cells = h3.grid_disk(center, k)
        cell_ints = ", ".join(str(int(c, 16)) for c in cells)
        clauses.append(f"h3_9 IN ({cell_ints})")

    return f"WHERE {' AND '.join(clauses)}" if clauses else ""



def _print_load_summary(results: list[LoadResult]) -> None:
    table = Table(title="Load Summary", show_lines=False)
    table.add_column("Table",      style="cyan", no_wrap=True)
    table.add_column("Rows",       justify="right")
    table.add_column("Start",      justify="right")
    table.add_column("End",        justify="right")
    table.add_column("Elapsed (s)", justify="right")
    for r in results:
        table.add_row(
            r.table,
            f"{r.row_count:,}",
            r.started_at.strftime("%H:%M:%S"),
            r.finished_at.strftime("%H:%M:%S"),
            f"{r.elapsed_s:.1f}",
        )
    _console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def schema(
    dims: Optional[int] = typer.Option(None,   "--dims",       help="Vector dimension (required for QBit/vsim tables)"),
    drop_first: bool     = typer.Option(False,  "--drop-first", help="Drop all tables before creating"),
) -> None:
    """Create (or recreate) all ClickHouse benchmark tables."""
    asyncio.run(_schema_async(dims, drop_first))


async def _schema_async(dims: int | None, drop_first: bool) -> None:
    client = await _make_client()
    try:
        if drop_first:
            with Status("Dropping tables..."):
                await drop_all_tables(client)
        if dims is not None:
            with Status(f"Ensuring tables for dims={dims}..."):
                await ensure_tables(client, dims=dims)
        else:
            with Status("Creating base tables (no dims — skipping QBit/vsim)..."):
                await create_all_tables(client)
    finally:
        await client.close()


@app.command()
def load(
    vectors_file: Optional[str] = typer.Option(None,         "--vectors-file",
        help="Path to .npy/.csv/.parquet with pre-computed vectors. "
             "Ignores --dims/--n-vectors/--dtype when provided."),
    vector_col:   str           = typer.Option("",           "--vector-col",
        help="Parquet column containing vectors (auto-detected if empty)."),
    dims:         int           = typer.Option(128,           "--dims",      help="Vector dimension (synthetic only)"),
    n_vectors:    int           = typer.Option(1_000,         "--n-vectors", help="Number of vectors (synthetic only)"),
    dtype:        str           = typer.Option("float32",     "--dtype",     help="float32 or float64 (synthetic only)"),
    n_queries:    int           = typer.Option(100,           "--n-queries", help="Number of held-out query vectors"),
    append:       bool          = typer.Option(False,         "--append",    help="Append to existing tables instead of dropping first"),
) -> None:
    """Load vectors into all ClickHouse tables.

    Two modes:
      --vectors-file  Load from your own .npy, .csv, or .parquet file.
      (no flag)       Generate synthetic random vectors (good for quick tests).

    Tables are dropped and recreated on every load by default to guarantee a
    clean slate. Use --append to add rows to existing tables instead.
    """
    asyncio.run(_load_async(dims, n_vectors, dtype, n_queries, vectors_file, vector_col, append))


async def _load_async(
    dims: int,
    n_vectors: int,
    dtype: str,
    n_queries: int,
    vectors_file: str | None = None,
    vector_col: str = "",
    append: bool = False,
) -> None:
    if vectors_file:
        from src.generate import load_vectors_file
        all_vecs = load_vectors_file(vectors_file, vector_col=vector_col)
        n_q = min(n_queries, max(1, len(all_vecs) // 5))
        db_vecs   = all_vecs[:-n_q]
        query_vecs = all_vecs[-n_q:]
        dims  = db_vecs.shape[1]
        dtype = "float64"
        _console.print(f"Using {len(db_vecs):,} db vectors + {n_q} query vectors (dims={dims})")
    else:
        _console.print(f"Generating {n_vectors:,} × {dims}-dim {dtype} vectors ({n_queries} held out)...")
        db_vecs, query_vecs = generate_vectors(n_vectors, dims, dtype=dtype, n_queries=n_queries)

    rows = vectors_to_rows(db_vecs)

    client = await _make_client()
    try:
        if not append:
            _console.print("[yellow]Dropping all tables for a clean slate (use --append to skip).[/yellow]")
            await drop_all_tables(client)
            await client.close()
            client = await _make_client()

        load_results = await load_all(client, db_vecs, rows)
    finally:
        await client.close()

    _save_query_state(query_vecs, dims, dtype)
    _console.print(f"\n[green]Query state saved ({len(query_vecs)} vectors → {QUERY_STATE_NPY})[/green]")
    _print_load_summary(load_results)


@app.command()
def bench(
    top_k:           int            = typer.Option(10,           "--top-k",          help="K for recall@K"),
    qbit_bits:       str            = typer.Option("4,8,16,32",  "--qbit-bits",       help="Comma-separated QBit bit depths"),
    export:          Optional[str]  = typer.Option(None,         "--export",          help="Export path (.json or .csv)"),
    filter_location: Optional[str]  = typer.Option(None,         "--filter-location", help="Filter by location name"),
    filter_after:    Optional[str]  = typer.Option(None,         "--filter-after",    help="Filter ts >= date (YYYY-MM-DD)"),
    filter_h3:       Optional[str]  = typer.Option(None,         "--filter-h3",       help="Filter by H3 ring: 'lat,lon,res,k'"),
) -> None:
    """Run benchmarks against pre-loaded vectors (run `load` first)."""
    asyncio.run(_bench_async(top_k, qbit_bits, export, filter_location, filter_after, filter_h3))


async def _bench_async(
    top_k: int,
    qbit_bits: str,
    export: str | None,
    filter_location: str | None,
    filter_after: str | None,
    filter_h3: str | None,
) -> None:
    query_vecs, dims, dtype = _load_query_state()
    bit_depths = [int(x.strip()) for x in qbit_bits.split(",")]
    where_clause = _build_where_clause(filter_location, filter_after, filter_h3)

    active_filters = [f for f in [filter_location, filter_after, filter_h3] if f]
    title = "Benchmark Results"
    if active_filters:
        title += f" [filters: {', '.join(active_filters)}]"

    _console.print(f"Query vectors: {query_vecs.shape}, dims={dims}, dtype={dtype}")

    # Guard: verify table dimension matches saved query state
    client_check = await _make_client()
    try:
        r = await client_check.query(
            f"SELECT length(vec) FROM {CLICKHOUSE_DATABASE}.vec_brute_f32 LIMIT 1"
        )
        if r.result_rows:
            table_dims = int(r.result_rows[0][0])
            if table_dims != dims:
                raise typer.BadParameter(
                    f"Dimension mismatch: query state has dims={dims} but "
                    f"vec_brute_f32 has dims={table_dims}. "
                    f"Run `load --dims {table_dims}` or `run --dims {table_dims}` first."
                )
    finally:
        await client_check.close()
    if where_clause:
        _console.print(f"Filter: [yellow]{where_clause}[/yellow]")

    client = await _make_client()
    try:
        with Status("Computing ground truth..."):
            ground_truth = await get_ground_truth(client, query_vecs, top_k, where_clause)

        with Status("Running benchmarks..."):
            results = await run_all(
                client, query_vecs, ground_truth,
                dims=dims, top_k=top_k,
                bit_depths=bit_depths, dtype=dtype,
                where_clause=where_clause,
            )
    finally:
        await client.close()

    print_results_table(results, title=title)

    if export:
        export_results(results, export)


@app.command()
def run(
    dims:      int           = typer.Option(128,          "--dims",      help="Vector dimension"),
    n_vectors: int           = typer.Option(1_000,        "--n-vectors", help="Number of DB vectors"),
    dtype:     str           = typer.Option("float32",    "--dtype",     help="float32 or float64"),
    n_queries: int           = typer.Option(100,          "--n-queries", help="Number of held-out query vectors"),
    top_k:     int           = typer.Option(10,           "--top-k",     help="K for recall@K"),
    qbit_bits: str           = typer.Option("4,8,16,32",  "--qbit-bits", help="Comma-separated QBit bit depths"),
    export:    Optional[str] = typer.Option(None,         "--export",    help="Export path (.json or .csv)"),
) -> None:
    """Convenience: load vectors then run benchmarks in one command."""
    asyncio.run(_run_async(dims, n_vectors, dtype, n_queries, top_k, qbit_bits, export))


async def _run_async(
    dims: int,
    n_vectors: int,
    dtype: str,
    n_queries: int,
    top_k: int,
    qbit_bits: str,
    export: str | None,
) -> None:
    # Drop and recreate tables for a clean slate before loading
    client = await _make_client()
    try:
        with Status("Resetting tables for clean run..."):
            await drop_all_tables(client)
            await ensure_tables(client, dims=dims)
    finally:
        await client.close()

    await _load_async(dims, n_vectors, dtype, n_queries)
    await _bench_async(top_k, qbit_bits, export, None, None, None)



@app.command("load-hf")
def load_hf(
    n:         int = typer.Option(500, "--n",         help="Number of images to download"),
    size:      int = typer.Option(8,   "--size",      help="Resize to size×size pixels (8=64-dim)"),
    n_queries: int = typer.Option(50,  "--n-queries", help="Held-out query images"),
) -> None:
    """Load real image descriptors from HuggingFace traffic-camera-norway-images."""
    asyncio.run(_load_hf_async(n, size, n_queries))


async def _load_hf_async(n: int, size: int, n_queries: int) -> None:
    from src.embed import load_hf_descriptors

    dims = size * size
    db_vecs, query_vecs, db_meta, _ = load_hf_descriptors(n, size, n_queries)
    _console.print(f"Descriptors ready: {db_vecs.shape}, dtype={db_vecs.dtype}")

    rows = vectors_to_rows(db_vecs, metadata=db_meta)

    client = await _make_client()
    try:
        load_results = await load_all(client, db_vecs, rows)
    finally:
        await client.close()

    _save_query_state(query_vecs, dims, "float64")
    _console.print(f"[green]Query state saved ({n_queries} vectors → {QUERY_STATE_NPY})[/green]")
    _print_load_summary(load_results)


if __name__ == "__main__":
    app()
