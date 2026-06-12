import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import clickhouse_connect
import numpy as np

from src.config import (
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE,
)
from src.schema import ensure_tables

DB = CLICKHOUSE_DATABASE

COLUMN_NAMES = ['id', 'ts', 'lat', 'lon', 'h3_9', 'location', 'vec']

ALL_TABLES = [
    'vec_brute_f32',
    'vec_brute_f64',
    'vec_qbit_f32',
    'vec_qbit_f64',
    'vec_vsim_f32',
    'vec_vsim_f16',
    'vec_vsim_i8',
]

# Tables that need float64 vec — all others use float32
_F64_TABLES = {'vec_brute_f64', 'vec_qbit_f64'}


@dataclass
class LoadResult:
    table: str
    row_count: int
    started_at: datetime   # naive UTC
    finished_at: datetime  # naive UTC
    elapsed_s: float


async def load_table(
    client,
    table: str,
    rows: list[dict],
    batch_size: int = 10_000,
) -> int:
    """Insert rows into table in batches. Returns total rows inserted."""
    total = len(rows)
    n_batches = (total + batch_size - 1) // batch_size
    inserted = 0

    for i in range(n_batches):
        batch = rows[i * batch_size : (i + 1) * batch_size]
        data = [[r[c] for c in COLUMN_NAMES] for r in batch]
        await client.insert(f'{DB}.{table}', data, column_names=COLUMN_NAMES)
        inserted += len(batch)
        print(f'    batch {i + 1}/{n_batches} — {inserted}/{total} rows')

    return inserted


async def load_all(
    client,
    db_vectors: np.ndarray,
    rows: list[dict],
    tables: list[str] | None = None,
    batch_size: int = 10_000,
) -> list[LoadResult]:
    """Load vectors into all (or selected) tables.

    Automatically calls ensure_tables() first so dimension-sensitive tables
    (QBit, vsim) are created or recreated to match db_vectors.shape[1].

    rows must come from vectors_to_rows(db_vectors) — vec is float32.
    For float64 tables, vec is recast from db_vectors on the fly.
    """
    dims = db_vectors.shape[1]
    print(f"Ensuring tables exist for dims={dims} ...")
    await ensure_tables(client, dims=dims)
    print()

    target_tables = tables if tables is not None else ALL_TABLES
    f64_vecs = db_vectors.astype(np.float64)
    results: list[LoadResult] = []

    for table in target_tables:
        # Build rows with the correct vec type for this table
        if table in _F64_TABLES:
            table_rows = [
                {**r, 'vec': f64_vecs[i].tolist()}
                for i, r in enumerate(rows)
            ]
        else:
            table_rows = rows  # vec is already float32 list

        started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        print(f'  loading {table} ...')
        count = await load_table(client, table, table_rows, batch_size=batch_size)
        finished_at = datetime.now(timezone.utc).replace(tzinfo=None)
        elapsed = (finished_at - started_at).total_seconds()

        print(
            f'  ✓ {table}: {count} rows | '
            f'{started_at:%H:%M:%S} → {finished_at:%H:%M:%S} ({elapsed:.1f}s)'
        )
        results.append(LoadResult(
            table=table,
            row_count=count,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_s=elapsed,
        ))

    return results


async def _main() -> None:
    from src.generate import generate_vectors, vectors_to_rows

    db_vectors, _ = generate_vectors(1000, 128)
    rows = vectors_to_rows(db_vectors)

    client = await clickhouse_connect.get_async_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD,
        database=DB, secure=True,
    )
    try:
        results = await load_all(client, db_vectors, rows, batch_size=1_000)
        print('\n=== Summary ===')
        for r in results:
            print(f'  {r.table:20s}  {r.row_count:>6} rows  {r.elapsed_s:.1f}s')
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(_main())
