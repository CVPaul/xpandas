#!/usr/bin/env python
"""bench_comparison.py -- Multi-library benchmark: xpandas vs pandas vs Polars."""
from __future__ import annotations
import sys
import time
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import torch
import numpy as np
import pandas as pd

# Load xpandas C++ ops
import xpandas  # noqa: F401

# Polars conditional import
type_has_polars = False
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# matplotlib conditional import
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

WARMUP = 3
REPEAT = 20

# Helper for timing
def _median_time(fn, warmup=WARMUP, repeat=REPEAT):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times.sort()
    mid = len(times) // 2
    if len(times) % 2 == 0:
        return (times[mid - 1] + times[mid]) / 2
    return times[mid]

@dataclass
class ComparisonResult:
    name: str
    n: int
    xpandas_us: float | None
    pandas_us: float | None
    polars_us: float | None
    speedup_vs_pandas: float | None
    speedup_vs_polars: float | None

# Data generators
def gen_float(N):
    return torch.randn(N, dtype=torch.double)
def gen_pos(N):
    return torch.rand(N, dtype=torch.double) + 0.01
def gen_group_keys(N):
    return torch.randint(0, 50, (N,), dtype=torch.long)
def gen_nan(N):
    arr = torch.randn(N, dtype=torch.double)
    arr[arr.abs() < 0.2] = float('nan')
    return arr
def gen_two(N):
    a = torch.randn(N, dtype=torch.double)
    b = torch.randn(N, dtype=torch.double)
    return a, b

def _to_pandas(t):
    return pd.Series(t.numpy())
def _to_polars(t):
    return pl.Series(t.numpy()) if HAS_POLARS else None

# Benchmark functions
def bench_rolling_mean(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.rolling_mean(x, 20)
    def pandas_fn():
        pd_x.rolling(20).mean()
    def polars_fn():
        if HAS_POLARS:
            pl_x.rolling_mean(window_size=20)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("rolling_mean", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_rolling_sum(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.rolling_sum(x, 20)
    def pandas_fn():
        pd_x.rolling(20).sum()
    def polars_fn():
        if HAS_POLARS:
            pl_x.rolling_sum(window_size=20)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("rolling_sum", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_groupby_sum(N):
    keys = gen_group_keys(N)
    vals = gen_float(N)
    pd_df = pd.DataFrame({"k": keys.numpy(), "v": vals.numpy()})
    if HAS_POLARS:
        pl_df = pl.DataFrame({"k": keys.numpy(), "v": vals.numpy()})
    def xpandas_fn():
        torch.ops.xpandas.groupby_sum(keys, vals)
    def pandas_fn():
        pd_df.groupby("k")["v"].sum()
    def polars_fn():
        if HAS_POLARS:
            pl_df.group_by("k").agg(pl.col("v").sum())
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("groupby_sum", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_groupby_mean(N):
    keys = gen_group_keys(N)
    vals = gen_float(N)
    pd_df = pd.DataFrame({"k": keys.numpy(), "v": vals.numpy()})
    if HAS_POLARS:
        pl_df = pl.DataFrame({"k": keys.numpy(), "v": vals.numpy()})
    def xpandas_fn():
        torch.ops.xpandas.groupby_mean(keys, vals)
    def pandas_fn():
        pd_df.groupby("k")["v"].mean()
    def polars_fn():
        if HAS_POLARS:
            pl_df.group_by("k").agg(pl.col("v").mean())
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("groupby_mean", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_fillna(N):
    x = gen_nan(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.fillna(x, 0.0)
    def pandas_fn():
        pd_x.fillna(0.0)
    def polars_fn():
        if HAS_POLARS:
            pl_x.fill_nan(0.0)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("fillna", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_clip(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.clip(x, -1.0, 1.0)
    def pandas_fn():
        pd_x.clip(-1.0, 1.0)
    def polars_fn():
        if HAS_POLARS:
            pl_x.clip(-1.0, 1.0)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("clip", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_abs(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.abs_(x)
    def pandas_fn():
        pd_x.abs()
    def polars_fn():
        if HAS_POLARS:
            pl_x.abs()
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("abs_", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_zscore(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.zscore(x)
    def pandas_fn():
        (pd_x - pd_x.mean()) / pd_x.std()
    def polars_fn():
        if HAS_POLARS:
            s = pl_x; (s - s.mean()) / s.std()
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("zscore", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_pct_change(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.pct_change(x, 1)
    def pandas_fn():
        pd_x.pct_change()
    def polars_fn():
        if HAS_POLARS:
            pl_x.pct_change()
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("pct_change", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_shift(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.shift(x, 3)
    def pandas_fn():
        pd_x.shift(3)
    def polars_fn():
        if HAS_POLARS:
            pl_x.shift(3)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("shift", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_cumsum(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.cumsum(x)
    def pandas_fn():
        pd_x.cumsum()
    def polars_fn():
        if HAS_POLARS:
            pl_x.cum_sum()
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("cumsum", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_compare_gt(N):
    a, b = gen_two(N)
    pd_a = _to_pandas(a)
    pd_b = _to_pandas(b)
    pl_a = _to_polars(a)
    pl_b = _to_polars(b)
    def xpandas_fn():
        torch.ops.xpandas.compare_gt(a, b)
    def pandas_fn():
        pd_a > pd_b
    def polars_fn():
        if HAS_POLARS:
            pl_a > pl_b
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("compare_gt", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_to_datetime(N):
    x = gen_pos(N)
    pd_x = _to_pandas(x)
    def xpandas_fn():
        torch.ops.xpandas.to_datetime(x, "s")
    def pandas_fn():
        pd.to_datetime(pd_x, unit="s")
    # No polars equivalent
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    return ComparisonResult("to_datetime", N, xp_us, pd_us, None,
        pd_us/xp_us if pd_us else None, None)

def bench_ewm_mean(N):
    x = gen_float(N)
    pd_x = _to_pandas(x)
    pl_x = _to_polars(x)
    def xpandas_fn():
        torch.ops.xpandas.ewm_mean(x, 10)
    def pandas_fn():
        pd_x.ewm(span=10, adjust=False).mean()
    def polars_fn():
        if HAS_POLARS:
            pl_x.ewm_mean(span=10, adjust=False)
    xp_us = _median_time(xpandas_fn) * 1e6
    pd_us = _median_time(pandas_fn) * 1e6
    pl_us = _median_time(polars_fn) * 1e6 if HAS_POLARS else None
    return ComparisonResult("ewm_mean", N, xp_us, pd_us, pl_us,
        pd_us/xp_us if pd_us else None,
        pl_us/xp_us if pl_us else None)

def bench_breakout_signal(N):
    price = gen_float(N)
    high = gen_float(N)
    low = gen_float(N)
    def xpandas_fn():
        torch.ops.xpandas.breakout_signal(price, high, low)
    xp_us = _median_time(xpandas_fn) * 1e6
    return ComparisonResult("breakout_signal", N, xp_us, None, None, None, None)

ALL_BENCHMARKS = [
    bench_rolling_mean,
    bench_rolling_sum,
    bench_groupby_sum,
    bench_groupby_mean,
    bench_fillna,
    bench_clip,
    bench_abs,
    bench_zscore,
    bench_pct_change,
    bench_shift,
    bench_cumsum,
    bench_compare_gt,
    bench_to_datetime,
    bench_ewm_mean,
    bench_breakout_signal,
]

# Chart generation
def _generate_charts(results, chart_dir):
    if not HAS_MPL:
        print("[WARN] matplotlib not installed, skipping chart generation.", file=sys.stderr)
        return
    chart_dir.mkdir(parents=True, exist_ok=True)
    # Grouped bar chart by op (N=10000)
    by_op = [r for r in results if r.n == 10000]
    ops = [r.name for r in by_op]
    xp = [r.xpandas_us for r in by_op]
    pd_ = [r.pandas_us if r.pandas_us is not None else 0 for r in by_op]
    pl_ = [r.polars_us if r.polars_us is not None else 0 for r in by_op]
    x = np.arange(len(ops))
    width = 0.25
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-whitegrid')
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, xp, width, label='xpandas')
    ax.bar(x, pd_, width, label='pandas')
    ax.bar(x + width, pl_, width, label='polars')
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=30, ha='right')
    ax.set_ylabel('Median time (μs)')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('Benchmark by Operation (N=10,000)')
    plt.tight_layout()
    plt.savefig(chart_dir / 'comparison_by_op.png')
    plt.close(fig)
    # Line chart by data size for key ops
    key_ops = ['rolling_mean', 'groupby_sum', 'fillna', 'zscore']
    fig, ax = plt.subplots(figsize=(10, 6))
    for lib, color in [('xpandas_us', 'C0'), ('pandas_us', 'C1'), ('polars_us', 'C2')]:
        for op in key_ops:
            ys = [getattr(r, lib) for r in results if r.name == op]
            ns = [r.n for r in results if r.name == op]
            # Filter out None values
            valid = [(n, y) for n, y in zip(ns, ys) if y is not None]
            if valid:
                label = f'{op} ({lib.split("_")[0]})'
                ax.plot([v[0] for v in valid], [v[1] for v in valid], marker='o', label=label, color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Data size (N)')
    ax.set_ylabel('Median time (μs)')
    ax.set_title('Benchmark by Data Size')
    ax.legend()
    plt.tight_layout()
    plt.savefig(chart_dir / 'comparison_by_size.png')
    plt.close(fig)

# Output formatting
def _format_table(results, sizes):
    out = []
    out.append("Multi-Library Benchmark: xpandas vs pandas vs Polars")
    out.append("=" * 53)
    for N in sizes:
        out.append(f"N={N:,} | repeat={REPEAT} | warmup={WARMUP}\n")
        out.append("Op                   xpandas (μs)  pandas (μs)  polars (μs)  xp/pd     xp/pl")
        out.append("---------------------------------------------------------------------------------")
        for r in [x for x in results if x.n == N]:
            xp = f"{r.xpandas_us:.1f}" if r.xpandas_us is not None else "N/A"
            pd_ = f"{r.pandas_us:.1f}" if r.pandas_us is not None else "N/A"
            pl_ = f"{r.polars_us:.1f}" if r.polars_us is not None else "N/A"
            xp_pd = f"{r.speedup_vs_pandas:.1f}x" if r.speedup_vs_pandas is not None else ""
            xp_pl = f"{r.speedup_vs_polars:.1f}x" if r.speedup_vs_polars is not None else ""
            out.append(f"{r.name:<22}{xp:>10}  {pd_:>10}  {pl_:>10}  {xp_pd:>7}  {xp_pl:>7}")
        out.append("")
    return "\n".join(out)

def _format_json(results):
    arr = []
    for r in results:
        arr.append({
            "name": r.name,
            "n": r.n,
            "xpandas_us": round(r.xpandas_us, 2) if r.xpandas_us is not None else None,
            "pandas_us": round(r.pandas_us, 2) if r.pandas_us is not None else None,
            "polars_us": round(r.polars_us, 2) if r.polars_us is not None else None,
        })
    return json.dumps(arr, indent=2)

# Main benchmark runner
def run_benchmarks(sizes, filter_str, as_json, generate_charts):
    results = []
    for N in sizes:
        for bench in ALL_BENCHMARKS:
            name = bench.__name__.replace('bench_', '')
            if filter_str and filter_str not in name:
                continue
            res = bench(N)
            results.append(res)
    chart_dir = Path(__file__).parent / "charts"
    if generate_charts:
        _generate_charts(results, chart_dir)
    if as_json:
        print(_format_json(results))
    else:
        print(_format_table(results, sizes))

# CLI

def main():
    parser = argparse.ArgumentParser(description="Multi-library benchmark: xpandas vs pandas vs Polars")
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--no-chart', action='store_true', help='Skip chart generation')
    parser.add_argument('--filter', type=str, default='', help='Only run benchmarks matching pattern')
    parser.add_argument('--sizes', type=str, default='1000,10000,100000', help='Comma-separated data sizes')
    args = parser.parse_args()
    sizes = [int(x) for x in args.sizes.split(',') if x.strip()]
    run_benchmarks(sizes, args.filter, args.json, not args.no_chart)

if __name__ == '__main__':
    main()
