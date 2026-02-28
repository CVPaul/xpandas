"""
bench_wrappers.py -- Benchmark per-op Python wrapper dispatch overhead and end-to-end Alpha performance.

Usage:
    python benchmarks/bench_wrappers.py
    python benchmarks/bench_wrappers.py --json | python -m json.tool
    python benchmarks/bench_wrappers.py --filter Series

Compares:
  Part 1: Direct C++ op call vs Python wrapper (Series/DataFrame)
  Part 2: End-to-end Alpha (pandas vs xpandas wrappers)
"""
from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import dataclass
import torch
import pandas as pd
import xpandas  # noqa: F401

WARMUP = 5
REPEAT = 30

# Helpers

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
class WrapperResult:
    operation: str
    direct_us: float
    wrapper_us: float
    overhead_us: float
    overhead_pct: float

@dataclass
class E2EResult:
    size_label: str
    n_inst: int
    n_prices: int
    pandas_us: float
    xpandas_us: float
    speedup: float

# Data generators

def _make_float_tensor(n):
    return torch.randn(n, dtype=torch.double)

def _make_int_keys(n, n_groups=50):
    return torch.randint(0, n_groups, (n,), dtype=torch.long)

def _make_alpha_data(n_inst, n_prices):
    total = n_inst * n_prices
    inst = torch.arange(n_inst, dtype=torch.long).repeat_interleave(n_prices)
    price = torch.rand(total, dtype=torch.double) * 100 + 50
    return {"InstrumentID": inst, "price": price}

# Part 1: Per-op wrapper overhead benchmarks

def _wrapper_result(name, direct_fn, wrapper_fn, repeat):
    direct_s = _median_time(direct_fn, warmup=WARMUP, repeat=repeat)
    wrapper_s = _median_time(wrapper_fn, warmup=WARMUP, repeat=repeat)
    direct_us = direct_s * 1e6
    wrapper_us = wrapper_s * 1e6
    overhead_us = wrapper_us - direct_us
    overhead_pct = (overhead_us / direct_us * 100) if direct_us > 0 else float('inf')
    return WrapperResult(name, direct_us, wrapper_us, overhead_us, overhead_pct)

def bench_series_gt(n, repeat):
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    sa = xpandas.wrappers.Series(a)
    sb = xpandas.wrappers.Series(b)
    def direct():
        torch.ops.xpandas.compare_gt(a, b)
    def wrapper():
        sa > sb
    return _wrapper_result("Series.__gt__", direct, wrapper, repeat)

def bench_series_lt(n, repeat):
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    sa = xpandas.wrappers.Series(a)
    sb = xpandas.wrappers.Series(b)
    def direct():
        torch.ops.xpandas.compare_lt(a, b)
    def wrapper():
        sa < sb
    return _wrapper_result("Series.__lt__", direct, wrapper, repeat)

def bench_series_sub(n, repeat):
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    sa = xpandas.wrappers.Series(a)
    sb = xpandas.wrappers.Series(b)
    def direct():
        a - b
    def wrapper():
        sa - sb
    return _wrapper_result("Series.__sub__", direct, wrapper, repeat)

def bench_series_astype_float(n, repeat):
    a = _make_float_tensor(n)
    b = _make_float_tensor(n)
    mask = torch.ops.xpandas.compare_gt(a, b)
    s_mask = xpandas.wrappers.Series(mask)
    def direct():
        torch.ops.xpandas.bool_to_float(mask)
    def wrapper():
        s_mask.astype(float)
    return _wrapper_result("Series.astype(float)", direct, wrapper, repeat)

def bench_dataframe_getattr(n, repeat):
    data = {"price": _make_float_tensor(n)}
    df = xpandas.wrappers.DataFrame(data)
    def direct():
        data["price"]
    def wrapper():
        df.price
    return _wrapper_result("DataFrame.__getattr__", direct, wrapper, repeat)

def bench_groupby_ohlc_chain(n, repeat):
    data = _make_alpha_data(50, n // 50)
    key = data["InstrumentID"]
    val = data["price"]
    def direct():
        torch.ops.xpandas.groupby_resample_ohlc(key, val)
    def wrapper():
        df = xpandas.wrappers.DataFrame(data)
        gs = df.groupby("InstrumentID")
        gs["price"].resample("1D").first()
        gs["price"].resample("1D").max()
        gs["price"].resample("1D").min()
        gs["price"].resample("1D").last()
    return _wrapper_result("GroupBy→OHLC chain", direct, wrapper, repeat)

def bench_ohlc4_cached(n, repeat):
    data = _make_alpha_data(50, n // 50)
    key = data["InstrumentID"]
    val = data["price"]
    def direct():
        for _ in range(4):
            torch.ops.xpandas.groupby_resample_ohlc(key, val)
    def wrapper():
        rs = xpandas.wrappers.DataFrame(data).groupby("InstrumentID")["price"].resample("1D")
        rs.first(); rs.max(); rs.min(); rs.last()
    return _wrapper_result("OHLC×4 cached", direct, wrapper, repeat)

ALL_WRAPPER_BENCHMARKS = [
    bench_series_gt,
    bench_series_lt,
    bench_series_sub,
    bench_series_astype_float,
    bench_dataframe_getattr,
    bench_groupby_ohlc_chain,
    bench_ohlc4_cached,
]

# Part 2: End-to-end Alpha benchmarks

class _XpandasAlpha:
    def __init__(self): self.freq = "1s"
    def on_bod(self, timestamp, data):
        from xpandas.wrappers import DataFrame
        df = DataFrame(data)
        gs = df.groupby("InstrumentID")
        o = gs["price"].resample("1D").first()
        h = gs["price"].resample("1D").max()
        l = gs["price"].resample("1D").min()
        c = gs["price"].resample("1D").last()
        self.hist = DataFrame({"inst": o.index.get_level_values("InstrumentID"),
            "open": o.values, "high": h.values, "low": l.values, "close": c.values})
    def forward(self, timestamp, data):
        from xpandas.wrappers import DataFrame
        df = DataFrame(data)
        long_ = df.price > self.hist.high
        short = df.price < self.hist.low
        return long_.astype(float) - short.astype(float)

class _PandasAlpha:
    def __init__(self): self.freq = "1s"
    def on_bod(self, timestamp, data):
        import pandas as pd
        df = pd.DataFrame({"InstrumentID": data["InstrumentID"].numpy(), "price": data["price"].numpy()})
        gs = df.groupby("InstrumentID")["price"]
        o = gs.first(); h = gs.max(); l = gs.min(); c = gs.last()
        self.hist = pd.DataFrame({"inst": o.index, "open": o.values, "high": h.values,
            "low": l.values, "close": c.values})
    def forward(self, timestamp, data):
        import pandas as pd
        price = pd.Series(data["price"].numpy())
        long_ = price > self.hist["high"].values
        short = price < self.hist["low"].values
        return (long_.astype(float) - short.astype(float)).values

E2E_SIZES = [
    (10,  50,  "Small (10×50)"),
    (50,  100, "Medium (50×100)"),
    (200, 500, "Large (200×500)"),
]

def bench_e2e(repeat, filter_str=None):
    results = []
    for n_inst, n_prices, label in E2E_SIZES:
        if filter_str and filter_str.lower() not in label.lower():
            continue
        bod_data = _make_alpha_data(n_inst, n_prices)
        fwd_data = {"price": torch.rand(n_inst, dtype=torch.double) * 100 + 50}
        # pandas
        pa = _PandasAlpha()
        def pandas_fn():
            pa.on_bod(0, bod_data)
            pa.forward(0, fwd_data)
        pandas_s = _median_time(pandas_fn, warmup=WARMUP, repeat=repeat)
        # xpandas
        xa = _XpandasAlpha()
        def xpandas_fn():
            xa.on_bod(0, bod_data)
            xa.forward(0, fwd_data)
        xpandas_s = _median_time(xpandas_fn, warmup=WARMUP, repeat=repeat)
        pandas_us = pandas_s * 1e6
        xpandas_us = xpandas_s * 1e6
        speedup = pandas_us / xpandas_us if xpandas_us > 0 else float('inf')
        results.append(E2EResult(label, n_inst, n_prices, pandas_us, xpandas_us, speedup))
    return results

# Runner

def run_benchmarks(n, repeat, filter_str, as_json):
    # Part 1
    wrapper_results = []
    for bench_fn in ALL_WRAPPER_BENCHMARKS:
        if filter_str and filter_str.lower() not in bench_fn.__name__.lower() and filter_str.lower() not in bench_fn.__name__.replace('_', ' ').lower():
            continue
        r = bench_fn(n, repeat)
        wrapper_results.append(r)
    # Part 2
    e2e_results = bench_e2e(repeat, filter_str)
    if as_json:
        out = {
            "wrapper": [
                {
                    "operation": r.operation,
                    "direct_us": round(r.direct_us, 2),
                    "wrapper_us": round(r.wrapper_us, 2),
                    "overhead_us": round(r.overhead_us, 2),
                    "overhead_pct": round(r.overhead_pct, 2),
                } for r in wrapper_results
            ],
            "e2e": [
                {
                    "size_label": r.size_label,
                    "n_inst": r.n_inst,
                    "n_prices": r.n_prices,
                    "pandas_us": round(r.pandas_us, 2),
                    "xpandas_us": round(r.xpandas_us, 2),
                    "speedup": round(r.speedup, 2),
                } for r in e2e_results
            ]
        }
        print(json.dumps(out, indent=2))
        return
    # Table output
    print("\nPart 1: Per-op Python wrapper dispatch overhead\n" + "="*55)
    print(f"{'Operation':<25} {'Direct (μs)':>12} {'Wrapper (μs)':>14} {'Overhead (μs)':>14} {'Overhead (%)':>13}")
    print("-"*55)
    for r in wrapper_results:
        print(f"{r.operation:<25} {r.direct_us:>12.1f} {r.wrapper_us:>14.1f} {r.overhead_us:>14.1f} {r.overhead_pct:>13.2f}")
    if wrapper_results:
        avg_overhead = sum(r.overhead_us for r in wrapper_results) / len(wrapper_results)
        avg_pct = sum(r.overhead_pct for r in wrapper_results) / len(wrapper_results)
        print("-"*55)
        print(f"{'Average overhead:':<25} {'':>12} {'':>14} {avg_overhead:>14.1f} {avg_pct:>13.2f}")
    print()
    print("Part 2: End-to-end Alpha performance\n" + "="*55)
    print(f"{'Size':<15} {'Instruments':>10} {'Pandas (μs)':>14} {'xpandas (μs)':>14} {'Speedup':>10}")
    print("-"*55)
    for r in e2e_results:
        print(f"{r.size_label:<15} {r.n_inst:>10} {r.pandas_us:>14.1f} {r.xpandas_us:>14.1f} {r.speedup:>10.2f}")
    if e2e_results:
        geomean = math.exp(sum(math.log(r.speedup) for r in e2e_results) / len(e2e_results))
        print("-"*55)
        print(f"{'Geometric mean speedup:':<15} {'':>10} {'':>14} {'':>14} {geomean:>10.2f}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Benchmark Python wrapper overhead and Alpha E2E performance")
    parser.add_argument("--n", type=int, default=10000, help="Number of elements for per-op benchmarks (default: 10000)")
    parser.add_argument("--repeat", type=int, default=30, help="Number of repeats (default: 30)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of tables")
    parser.add_argument("--filter", type=str, default=None, help="Filter benchmarks by substring")
    args = parser.parse_args()
    run_benchmarks(args.n, args.repeat, args.filter, args.json)

if __name__ == "__main__":
    main()
