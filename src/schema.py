import asyncio
import re
import clickhouse_connect
from src.config import (
    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
    CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE,
)

_SHARED_COLS = """
    id       UInt64,
    ts       DateTime               DEFAULT now(),
    lat      Float64                DEFAULT 0,
    lon      Float64                DEFAULT 0,
    h3_9     UInt64                 DEFAULT 0,
    location LowCardinality(String) DEFAULT ''
"""

_ENGINE = "ENGINE = MergeTree() PARTITION BY toYYYYMM(ts) ORDER BY id"

DB = CLICKHOUSE_DATABASE

# Tables whose vec column type embeds the dimension and must be recreated on change
_DIM_SENSITIVE = ["vec_qbit_f32", "vec_qbit_f64", "vec_vsim_f32", "vec_vsim_f16", "vec_vsim_i8"]


def _vsim_ddl(quant: str, dims: int, m: int = 16, ef_c: int = 100) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {DB}.vec_vsim_{quant} (
            {_SHARED_COLS},
            vec Array(Float32),
            INDEX idx_vec vec
                TYPE vector_similarity('hnsw', 'cosineDistance', {dims}, '{quant}', {m}, {ef_c})
                GRANULARITY 2
        ) {_ENGINE}
    """


def _qbit_table_specs(dims: int) -> dict[str, tuple[str, dict]]:
    """QBit(FloatXX, dims) — second param IS the vector dimension."""
    return {
        "vec_qbit_f32": (
            f"CREATE TABLE IF NOT EXISTS {DB}.vec_qbit_f32 ("
            f"{_SHARED_COLS}, vec QBit(Float32, {dims})) {_ENGINE}",
            {"allow_experimental_qbit_type": 1},
        ),
        "vec_qbit_f64": (
            f"CREATE TABLE IF NOT EXISTS {DB}.vec_qbit_f64 ("
            f"{_SHARED_COLS}, vec QBit(Float64, {dims})) {_ENGINE}",
            {"allow_experimental_qbit_type": 1},
        ),
    }


def _vsim_table_specs(dims: int) -> dict[str, tuple[str, dict]]:
    return {
        "vec_vsim_f32": (_vsim_ddl("f32", dims), {}),
        "vec_vsim_f16": (_vsim_ddl("f16", dims), {}),
        "vec_vsim_i8":  (_vsim_ddl("i8",  dims), {}),
    }


def _base_table_specs() -> dict[str, tuple[str, dict]]:
    """Array(Float32/64) tables — accept any dimension, never need recreation."""
    return {
        "vec_brute_f32": (
            f"CREATE TABLE IF NOT EXISTS {DB}.vec_brute_f32 ("
            f"{_SHARED_COLS}, vec Array(Float32)) {_ENGINE}",
            {},
        ),
        "vec_brute_f64": (
            f"CREATE TABLE IF NOT EXISTS {DB}.vec_brute_f64 ("
            f"{_SHARED_COLS}, vec Array(Float64)) {_ENGINE}",
            {},
        ),
    }


async def get_table_dims(client, table: str) -> int | None:
    """Return the vector dimension the table was created with, or None if it doesn't exist.

    Reads QBit dimension from the column type string: QBit(Float32, 128) → 128.
    Reads vsim dimension from the index type string: vector_similarity(..., 128, ...) → 128.
    """
    try:
        if table.startswith("vec_qbit"):
            r = await client.query(
                "SELECT type FROM system.columns "
                "WHERE database={db:String} AND table={t:String} AND name='vec'",
                parameters={"db": DB, "t": table},
            )
            if not r.result_rows:
                return None
            # e.g. "QBit(Float32, 768)"
            m = re.search(r"QBit\(\w+,\s*(\d+)\)", r.result_rows[0][0])
            return int(m.group(1)) if m else None

        elif table.startswith("vec_vsim"):
            r = await client.query(
                "SELECT type_full FROM system.data_skipping_indices "
                "WHERE database={db:String} AND table={t:String} AND name='idx_vec'",
                parameters={"db": DB, "t": table},
            )
            if not r.result_rows:
                return None
            # e.g. "vector_similarity('hnsw', 'cosineDistance', 768, 'f16', 16, 100)"
            m = re.search(r"cosineDistance',\s*(\d+)", r.result_rows[0][0])
            return int(m.group(1)) if m else None

    except Exception:
        return None

    return None


async def ensure_tables(client, dims: int) -> dict[str, str]:
    """Create or recreate tables so all are ready for `dims`-dimensional vectors.

    - brute tables: created once, never dropped (Array accepts any dim)
    - QBit / vsim tables: recreated if they don't exist or have a different dimension

    Returns {table_name: "created" | "ok" | "recreated" | "failed"}.
    """
    status: dict[str, str] = {}

    # --- brute tables: create if missing, never touch if they exist ---
    for name, (ddl, settings) in _base_table_specs().items():
        try:
            await client.command(ddl, settings=settings or None)
            status[name] = "ok"
        except Exception as exc:
            print(f"  ✗ {name}: {str(exc)[:120]}")
            status[name] = "failed"

    # --- dim-sensitive tables: check current dim, recreate if wrong ---
    dim_specs = {**_qbit_table_specs(dims), **_vsim_table_specs(dims)}
    for name, (ddl, settings) in dim_specs.items():
        current_dims = await get_table_dims(client, name)

        if current_dims == dims:
            status[name] = "ok"
            continue

        if current_dims is not None:
            # Exists but wrong dimension — drop it
            print(f"  ↻ {name}: dim {current_dims} → {dims}, recreating")
            await client.command(f"DROP TABLE IF EXISTS {DB}.{name}")
            action = "recreated"
        else:
            action = "created"

        # Strip IF NOT EXISTS so the create fails loudly if something went wrong
        create_ddl = ddl.replace("IF NOT EXISTS ", "")
        try:
            await client.command(create_ddl, settings=settings or None)
            status[name] = action
        except Exception as exc:
            print(f"  ✗ {name}: {str(exc)[:120]}")
            status[name] = "failed"

    created   = [n for n, s in status.items() if s == "created"]
    recreated = [n for n, s in status.items() if s == "recreated"]
    ok        = [n for n, s in status.items() if s == "ok"]
    failed    = [n for n, s in status.items() if s == "failed"]

    if created:   print(f"  ✓ created:   {created}")
    if recreated: print(f"  ↻ recreated: {recreated}")
    if ok:        print(f"  · unchanged: {ok}")
    if failed:    print(f"  ✗ failed:    {failed}")

    return status


async def create_all_tables(
    client, drop_first: bool = False, dims: int | None = None
) -> dict[str, bool]:
    """Explicit create (optionally drop first). For manual / CLI use.

    Prefer ensure_tables() for programmatic use — it handles dim changes automatically.
    """
    if drop_first:
        await drop_all_tables(client)

    specs = _base_table_specs()
    if dims is not None:
        specs.update(_qbit_table_specs(dims))
        specs.update(_vsim_table_specs(dims))
    else:
        print("  note: skipping QBit and vec_vsim_* tables (pass dims= to create them)")

    results: dict[str, bool] = {}
    for name, (ddl, settings) in specs.items():
        try:
            await client.command(ddl, settings=settings or None)
            results[name] = True
        except Exception as exc:
            print(f"  ✗ {name}: {str(exc)[:120]}")
            results[name] = False

    created = [n for n, ok in results.items() if ok]
    skipped = [n for n, ok in results.items() if not ok]
    print(f"\n✓ created ({len(created)}): {created}")
    if skipped:
        print(f"✗ skipped ({len(skipped)}): {skipped}")
    return results


async def drop_all_tables(client) -> None:
    """Drop all benchmark tables (IF EXISTS)."""
    for name in list(_base_table_specs().keys()) + _DIM_SENSITIVE:
        try:
            await client.command(f"DROP TABLE IF EXISTS {DB}.{name}")
            print(f"  dropped {name}")
        except Exception as exc:
            print(f"  could not drop {name}: {exc}")


async def _main() -> None:
    client = await clickhouse_connect.get_async_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD,
        database=DB, secure=True,
    )
    try:
        print(f"Connected to {CLICKHOUSE_HOST} / {DB}")
        await ensure_tables(client, dims=128)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_main())
