#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_ops.py -- Benchmark xpandas C++ ops vs pandas equivalents.

Usage:
    python benchmarks/bench_ops.py                 # default N=10000
    python benchmarks/bench_ops.py --n 100000      # larger dataset
    python benchmarks/bench_ops.py --json           # machine-readable JSON output
    python benchmarks/bench_ops.py --filter rolling # only run benchmarks matching "rolling"

Each benchmark:
  1. Prepares matching data for both xpandas (Tensors) and pandas (Series/DataFrame).
  2. Runs each implementation `repeat` times, takes the median wall-clock time.
  3. Reports speedup = pandas_time / xpandas_time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

# Load xpandas C++ ops
import xpandas  # noqa: F401

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WARMUP = 3
REPEAT = 20


def _median_time(fn: Callable[[], object], warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    """Return median execution time (seconds) of `fn`."""
    for _ in range(warmup):
        fn()
    times: list[float] = []
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
class BenchResult:
    name: str
    pandas_us: float
    xpandas_us: float
    speedup: float
    n: int


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_float_tensor(n: int) -> torch.Tensor:
    return torch.randn(n, dtype=torch.double)


def _make_positive_tensor(n: int) -> torch.Tensor:
    return torch.rand(n, dtype=torch.double) + 0.01  # avoid zero / negative


def _make_int_keys(n: int, n_groups: int = 50) -> torch.Tensor:
    return torch.randint(0, n_groups, (n,), dtype=torch.long)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def bench_lookup(n: int) -> BenchResult:
    t = _make_float_tensor(n)
    table = {"col_a": t, "col_b": t.clone()}
    df = pd.DataFrame({"col_a": t.numpy(), "col_b": t.numpy()})

    def xp():
        torch.ops.xpandas.lookup(table, "col_a")

    def pd_fn():
        df["col_a"]

    return _result("lookup", n, xp, pd_fn)


def bench_groupby_sum(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_sum(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).sum()

    return _result("groupby_sum", n, xp, pd_fn)


def bench_groupby_mean(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_mean(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).mean()

    return _result("groupby_mean", n, xp, pd_fn)


def bench_groupby_count(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_count(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).count()

    return _result("groupby_count", n, xp, pd_fn)


def bench_groupby_std(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_std(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).std()

    return _result("groupby_std", n, xp, pd_fn)


def bench_groupby_min(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_min(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).min()

    return _result("groupby_min", n, xp, pd_fn)


def bench_groupby_max(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_max(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).max()

    return _result("groupby_max", n, xp, pd_fn)


def bench_groupby_first(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_first(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).first()

    return _result("groupby_first", n, xp, pd_fn)


def bench_groupby_last(n: int) -> BenchResult:
    keys = _make_int_keys(n)
    vals = _make_float_tensor(n)
    s_keys = pd.Series(keys.numpy())
    s_vals = pd.Series(vals.numpy())

    def xp():
        torch.ops.xpandas.groupby_last(keys, vals)

    def pd_fn():
        s_vals.groupby(s_keys).last()

    return _result("groupby_last", n, xp, pd_fn)


def bench_compare_gt(n: int) -> BenchResult:
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    sa = pd.Series(a.numpy())
    sb = pd.Series(b.numpy())

    def xp():
        torch.ops.xpandas.compare_gt(a, b)

    def pd_fn():
        sa > sb

    return _result("compare_gt", n, xp, pd_fn)


def bench_compare_lt(n: int) -> BenchResult:
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    sa = pd.Series(a.numpy())
    sb = pd.Series(b.numpy())

    def xp():
        torch.ops.xpandas.compare_lt(a, b)

    def pd_fn():
        sa < sb

    return _result("compare_lt", n, xp, pd_fn)


def bench_bool_to_float(n: int) -> BenchResult:
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    mask = torch.ops.xpandas.compare_gt(a, b)
    s_mask = pd.Series(mask.numpy().astype(bool))

    def xp():
        torch.ops.xpandas.bool_to_float(mask)

    def pd_fn():
        s_mask.astype(float)

    return _result("bool_to_float", n, xp, pd_fn)


def bench_rank(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.rank(x)

    def pd_fn():
        s.rank(method="average")

    return _result("rank", n, xp, pd_fn)


def bench_rolling_sum(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    w = 20

    def xp():
        torch.ops.xpandas.rolling_sum(x, w)

    def pd_fn():
        s.rolling(w).sum()

    return _result("rolling_sum", n, xp, pd_fn)


def bench_rolling_mean(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    w = 20

    def xp():
        torch.ops.xpandas.rolling_mean(x, w)

    def pd_fn():
        s.rolling(w).mean()

    return _result("rolling_mean", n, xp, pd_fn)


def bench_rolling_std(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    w = 20

    def xp():
        torch.ops.xpandas.rolling_std(x, w)

    def pd_fn():
        s.rolling(w).std()

    return _result("rolling_std", n, xp, pd_fn)


def bench_rolling_min(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    w = 20

    def xp():
        torch.ops.xpandas.rolling_min(x, w)

    def pd_fn():
        s.rolling(w).min()

    return _result("rolling_min", n, xp, pd_fn)


def bench_rolling_max(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    w = 20

    def xp():
        torch.ops.xpandas.rolling_max(x, w)

    def pd_fn():
        s.rolling(w).max()

    return _result("rolling_max", n, xp, pd_fn)


def bench_shift(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.shift(x, 5)

    def pd_fn():
        s.shift(5)

    return _result("shift", n, xp, pd_fn)


def bench_fillna(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    x[::3] = float("nan")
    s = pd.Series(x.numpy().copy())

    def xp():
        torch.ops.xpandas.fillna(x, 0.0)

    def pd_fn():
        s.fillna(0.0)

    return _result("fillna", n, xp, pd_fn)


def bench_where(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    other = _make_float_tensor(n)
    cond = torch.ops.xpandas.compare_gt(x, torch.zeros(n, dtype=torch.double))
    s_x = pd.Series(x.numpy())
    s_other = pd.Series(other.numpy())
    s_cond = pd.Series(cond.numpy().astype(bool))

    def xp():
        torch.ops.xpandas.where_(cond, x, other)

    def pd_fn():
        s_x.where(s_cond, s_other)

    return _result("where_", n, xp, pd_fn)


def bench_masked_fill(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    mask = (x > 0)  # bool tensor
    s_x = pd.Series(x.numpy().copy())
    s_mask = pd.Series(mask.numpy())

    def xp():
        torch.ops.xpandas.masked_fill(x, mask, -1.0)

    def pd_fn():
        s_x.mask(s_mask, -1.0)

    return _result("masked_fill", n, xp, pd_fn)


def bench_pct_change(n: int) -> BenchResult:
    x = _make_positive_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.pct_change(x, 1)

    def pd_fn():
        s.pct_change(1)

    return _result("pct_change", n, xp, pd_fn)


def bench_cumsum(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.cumsum(x)

    def pd_fn():
        s.cumsum()

    return _result("cumsum", n, xp, pd_fn)


def bench_cumprod(n: int) -> BenchResult:
    # Use values near 1.0 to avoid overflow
    x = torch.ones(n, dtype=torch.double) + torch.randn(n, dtype=torch.double) * 0.001
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.cumprod(x)

    def pd_fn():
        s.cumprod()

    return _result("cumprod", n, xp, pd_fn)


def bench_clip(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.clip(x, -1.0, 1.0)

    def pd_fn():
        s.clip(-1.0, 1.0)

    return _result("clip", n, xp, pd_fn)


def bench_abs(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.abs_(x)

    def pd_fn():
        s.abs()

    return _result("abs_", n, xp, pd_fn)


def bench_log(n: int) -> BenchResult:
    x = _make_positive_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.log_(x)

    def pd_fn():
        np.log(s)

    return _result("log_", n, xp, pd_fn)


def bench_zscore(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.zscore(x)

    def pd_fn():
        (s - s.mean()) / s.std()

    return _result("zscore", n, xp, pd_fn)


def bench_ewm_mean(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    s = pd.Series(x.numpy())
    span = 10

    def xp():
        torch.ops.xpandas.ewm_mean(x, span)

    def pd_fn():
        s.ewm(span=span, adjust=False).mean()

    return _result("ewm_mean", n, xp, pd_fn)


def bench_sort_by(n: int) -> BenchResult:
    x = _make_float_tensor(n)
    ids = torch.arange(n, dtype=torch.long)
    table = {"value": x, "id": ids}
    df = pd.DataFrame({"value": x.numpy(), "id": ids.numpy()})

    def xp():
        torch.ops.xpandas.sort_by(table, "value", True)

    def pd_fn():
        df.sort_values("value", ascending=True)

    return _result("sort_by", n, xp, pd_fn)


def bench_to_datetime(n: int) -> BenchResult:
    # Epoch seconds
    x = torch.arange(0, n, dtype=torch.double) + 1_600_000_000.0
    s = pd.Series(x.numpy())

    def xp():
        torch.ops.xpandas.to_datetime(x, "s")

    def pd_fn():
        pd.to_datetime(s, unit="s")

    return _result("to_datetime", n, xp, pd_fn)


def bench_dt_floor(n: int) -> BenchResult:
    # Nanosecond timestamps
    base_ns = 1_600_000_000 * 1_000_000_000
    x = torch.arange(0, n, dtype=torch.long) * 1_000_000 + base_ns  # 1ms apart
    interval_ns = 60 * 1_000_000_000  # 1 minute
    dt_index = pd.to_datetime(x.numpy(), unit="ns")

    def xp():
        torch.ops.xpandas.dt_floor(x, interval_ns)

    def pd_fn():
        dt_index.floor("min")

    return _result("dt_floor", n, xp, pd_fn)


def bench_breakout_signal(n: int) -> BenchResult:
    price = _make_float_tensor(n)
    high = price + 0.5
    low = price - 0.5
    s_price = pd.Series(price.numpy())
    s_high = pd.Series(high.numpy())
    s_low = pd.Series(low.numpy())

    def xp():
        torch.ops.xpandas.breakout_signal(price, high, low)

    def pd_fn():
        (s_price > s_high).astype(float) - (s_price < s_low).astype(float)

    return _result("breakout_signal", n, xp, pd_fn)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _result(name: str, n: int, xp_fn: Callable, pd_fn: Callable) -> BenchResult:
    t_xp = _median_time(xp_fn)
    t_pd = _median_time(pd_fn)
    speedup = t_pd / t_xp if t_xp > 0 else float("inf")
    return BenchResult(
        name=name,
        pandas_us=t_pd * 1e6,
        xpandas_us=t_xp * 1e6,
        speedup=speedup,
        n=n,
    )


ALL_BENCHMARKS: List[Callable[[int], BenchResult]] = [
    bench_lookup,
    bench_groupby_sum,
    bench_groupby_mean,
    bench_groupby_count,
    bench_groupby_std,
    bench_groupby_min,
    bench_groupby_max,
    bench_groupby_first,
    bench_groupby_last,
    bench_compare_gt,
    bench_compare_lt,
    bench_bool_to_float,
    bench_rank,
    bench_rolling_sum,
    bench_rolling_mean,
    bench_rolling_std,
    bench_rolling_min,
    bench_rolling_max,
    bench_shift,
    bench_fillna,
    bench_where,
    bench_masked_fill,
    bench_pct_change,
    bench_cumsum,
    bench_cumprod,
    bench_clip,
    bench_abs,
    bench_log,
    bench_zscore,
    bench_ewm_mean,
    bench_sort_by,
    bench_to_datetime,
    bench_dt_floor,
    bench_breakout_signal,
]


def run_benchmarks(
    n: int = 10_000,
    filter_str: Optional[str] = None,
    as_json: bool = False,
) -> List[BenchResult]:
    benchmarks = ALL_BENCHMARKS
    if filter_str:
        benchmarks = [b for b in benchmarks if filter_str.lower() in b.__name__.lower()]

    results: List[BenchResult] = []

    if not as_json:
        header = f"{'Op':<25} {'pandas (us)':>12} {'xpandas (us)':>13} {'speedup':>9}"
        print(f"\nBenchmark: xpandas vs pandas  (N={n:,}, repeat={REPEAT}, warmup={WARMUP})")
        print("=" * len(header))
        print(header)
        print("-" * len(header))

    for bench_fn in benchmarks:
        r = bench_fn(n)
        results.append(r)
        if not as_json:
            arrow = ">>>" if r.speedup >= 2.0 else ">>" if r.speedup >= 1.0 else "<<"
            print(f"{r.name:<25} {r.pandas_us:>12.1f} {r.xpandas_us:>13.1f} {r.speedup:>8.2f}x {arrow}")

    if not as_json:
        # Summary
        speedups = [r.speedup for r in results]
        geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        faster = sum(1 for s in speedups if s >= 1.0)
        print("-" * len(header))
        print(f"{'Geometric mean speedup:':<25} {'':>12} {'':>13} {geomean:>8.2f}x")
        print(f"Faster in {faster}/{len(results)} ops")
        print()
    else:
        out = [
            {
                "name": r.name,
                "n": r.n,
                "pandas_us": round(r.pandas_us, 2),
                "xpandas_us": round(r.xpandas_us, 2),
                "speedup": round(r.speedup, 2),
            }
            for r in results
        ]
        print(json.dumps(out, indent=2))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark xpandas ops vs pandas")
    parser.add_argument("--n", type=int, default=10_000, help="Number of elements (default: 10000)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--filter", type=str, default=None, help="Only run benchmarks matching this string")
    args = parser.parse_args()

    run_benchmarks(n=args.n, filter_str=args.filter, as_json=args.json)


if __name__ == "__main__":
    main()
