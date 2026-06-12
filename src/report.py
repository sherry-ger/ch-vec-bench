import json
from dataclasses import asdict

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.benchmark import BenchmarkResult

_console = Console()


def print_results_table(
    results: list[BenchmarkResult],
    title: str = "Benchmark Results",
) -> None:
    wide = Console(width=160)   # force wide output regardless of terminal size
    table = Table(title=title, show_lines=False)
    table.add_column("Strategy",    style="bold cyan", no_wrap=True)
    table.add_column("Dims",        justify="right",   no_wrap=True)
    table.add_column("N",           justify="right",   no_wrap=True)
    table.add_column("p50 ms",      justify="right",   no_wrap=True)
    table.add_column("p95 ms",      justify="right",   no_wrap=True)
    table.add_column("Recall",      justify="right",   no_wrap=True)
    table.add_column("Comp (MB)",   justify="right",   no_wrap=True)
    table.add_column("Raw (MB)",    justify="right",   no_wrap=True)
    table.add_column("Ratio",       justify="right",   no_wrap=True)

    for r in sorted(results, key=lambda x: x.compression_ratio, reverse=True):
        if r.recall_at_k >= 0.9:
            recall_str = f"[green]{r.recall_at_k:.3f}[/green]"
        elif r.recall_at_k >= 0.7:
            recall_str = f"[yellow]{r.recall_at_k:.3f}[/yellow]"
        else:
            recall_str = f"[red]{r.recall_at_k:.3f}[/red]"

        table.add_row(
            r.strategy,
            str(r.dims),
            str(r.n_vectors),
            f"{r.median_latency_ms:.1f}",
            f"{r.p95_latency_ms:.1f}",
            recall_str,
            f"{r.compressed_bytes / 1e6:.3f}",
            f"{r.uncompressed_bytes / 1e6:.3f}",
            f"{r.compression_ratio:.2f}x",
        )

    wide.print(table)


def export_results(results: list[BenchmarkResult], path: str) -> None:
    rows = [asdict(r) for r in results]
    if path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(rows, f, indent=2)
    elif path.endswith(".csv"):
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported export format: {path} (use .json or .csv)")
    _console.print(f"[green]Results exported to {path}[/green]")
