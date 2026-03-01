# xpandas

**English** | [中文](docs/README_zh.md)

Pandas-like DataFrame operations implemented as **PyTorch custom ops**
(`TORCH_LIBRARY`), enabling `torch.jit.script` compilation and pure C++
inference via `torch::jit::load()`.

## Why?

Quantitative trading strategies are often prototyped in Python using pandas.
Deploying them to a low-latency C++ engine traditionally requires a full
rewrite. xpandas bridges this gap:

1. Replace `import pandas as pd` with `import xpandas as pd`
2. `torch.jit.script(model)` compiles the module to TorchScript
3. Load the `.pt` file in C++ -- no Python runtime needed

## Architecture

```
  Python side                        C++ side
  -----------                        --------
  import xpandas                     dlopen(libxpandas_ops.so)
  model = Alpha()                    auto m = torch::jit::load("alpha.pt")
  scripted = torch.jit.script(model) m.get_method("on_bod")({ts, data})
  scripted.save("alpha.pt")          auto sig = m.forward({ts, data})
```

**Data model:**

- A "DataFrame" is `Dict[str, Tensor]` (column name -> 1-D tensor)
- String columns are enum-encoded to `int64` tensors
- Numeric columns are `float64` tensors
- Each pandas-like operation is a registered `torch.ops.xpandas.*` op

## Project Structure

```
xpandas/
  __init__.py              # Package init, loads C++ extension
  ops_meta.py              # FakeTensor kernels (for torch.compile)
  csrc/ops/
    ops.h                  # Common header with op declarations
    register.cpp           # TORCH_LIBRARY schema + CPU dispatch
    groupby_resample_ohlc.cpp
    compare.cpp
    cast.cpp
    lookup.cpp
    breakout_signal.cpp
    rank.cpp               # Example op (see CONTRIBUTING.md)
    to_datetime.cpp        # to_datetime + dt_floor
    groupby_agg.cpp        # groupby_sum/mean/count/std
    groupby_minmax.cpp     # groupby_min/max/first/last
    rolling.cpp            # rolling_sum/mean/std
    rolling_minmax.cpp     # rolling_min/max (O(n) monotonic deque)
    shift.cpp              # shift (lag/lead)
    fillna.cpp             # fillna
    where.cpp              # where_, masked_fill
    pct_change.cpp         # pct_change
    cumulative.cpp         # cumsum, cumprod
    clip.cpp               # clip
    math_ops.cpp           # abs_, log_, zscore
    ewm.cpp                # ewm_mean
    sort.cpp               # sort_by
inference/
  main.cpp                 # Pure C++ inference driver
examples/
  alpha_original.py        # Original pandas-based Alpha (reference)
  alpha_ts.py              # TorchScript-compatible Alpha (breakout)
  alpha_vwap.py            # TorchScript VWAP mean-reversion Alpha
  alpha_momentum.py        # TorchScript momentum z-score Alpha
  trace_and_save.py        # Script + test + save alpha.pt
benchmarks/
  bench_ops.py             # xpandas vs pandas performance comparison
tests/
  test_ops.py              # Unit tests for each op (110 tests)
  test_alpha_e2e.py        # End-to-end scripting tests (10 tests)
```

## Quickstart

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.0
- A C++ compiler with C++17 support

### Install (Python)

```bash
pip install --no-build-isolation -e .
```

> **Note:** `--no-build-isolation` is required to ensure the C++ extension is
> compiled with the same ABI as your installed PyTorch.

### Run Tests

```bash
pytest tests/ -v
```

### Script and Save a Model

```bash
python examples/trace_and_save.py
# produces alpha.pt
```

### Build and Run C++ Inference

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
make -j

./alpha_infer ../alpha.pt ./libxpandas_ops.so
# Output: Signal: [+1.0, -1.0]
```

## Available Ops (35 total)

### DataFrame Utilities

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `lookup` | `(Dict(str, Tensor) table, str key) -> Tensor` | `df['col']` |
| `sort_by` | `(Dict(str, Tensor) table, str by, bool ascending) -> Dict(str, Tensor)` | `df.sort_values(by)` |

### Groupby / Aggregation

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `groupby_resample_ohlc` | `(Tensor key, Tensor value) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` | `df.groupby(key)[val].resample().{first,max,min,last}()` |
| `groupby_sum` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].sum()` |
| `groupby_mean` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].mean()` |
| `groupby_count` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].count()` |
| `groupby_std` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].std()` |
| `groupby_min` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].min()` |
| `groupby_max` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].max()` |
| `groupby_first` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].first()` |
| `groupby_last` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].last()` |

### Element-wise Comparison

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `compare_gt` | `(Tensor a, Tensor b) -> Tensor` | `series > series` |
| `compare_lt` | `(Tensor a, Tensor b) -> Tensor` | `series < series` |

### Type Casting

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `bool_to_float` | `(Tensor x) -> Tensor` | `series.astype(float)` |

### Fused Signals

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `breakout_signal` | `(Tensor price, Tensor high, Tensor low) -> Tensor` | `(price > high).float() - (price < low).float()` |

### Statistical

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `rank` | `(Tensor x) -> Tensor` | `series.rank(method='average')` |
| `zscore` | `(Tensor x) -> Tensor` | `(series - series.mean()) / series.std()` |

### Datetime

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `to_datetime` | `(Tensor epochs, str unit) -> Tensor` | `pd.to_datetime(series, unit=...)` |
| `dt_floor` | `(Tensor dt_ns, int interval_ns) -> Tensor` | `series.dt.floor(freq)` |

### Rolling Window

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `rolling_sum` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).sum()` |
| `rolling_mean` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).mean()` |
| `rolling_std` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).std()` |
| `rolling_min` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).min()` |
| `rolling_max` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).max()` |

### Shift / Lag

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `shift` | `(Tensor x, int periods) -> Tensor` | `series.shift(periods)` |

### NaN Handling

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `fillna` | `(Tensor x, float fill_value) -> Tensor` | `series.fillna(value)` |

### Conditional

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `where_` | `(Tensor cond, Tensor x, Tensor other) -> Tensor` | `series.where(cond, other)` |
| `masked_fill` | `(Tensor x, Tensor mask, float fill_value) -> Tensor` | `series.mask(mask, value)` |

### Percentage Change

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `pct_change` | `(Tensor x, int periods) -> Tensor` | `series.pct_change(periods)` |

### Cumulative

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `cumsum` | `(Tensor x) -> Tensor` | `series.cumsum()` |
| `cumprod` | `(Tensor x) -> Tensor` | `series.cumprod()` |

### Clipping

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `clip` | `(Tensor x, float lower, float upper) -> Tensor` | `series.clip(lower, upper)` |

### Math

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `abs_` | `(Tensor x) -> Tensor` | `series.abs()` |
| `log_` | `(Tensor x) -> Tensor` | `np.log(series)` |

### Exponential Weighted

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `ewm_mean` | `(Tensor x, int span) -> Tensor` | `series.ewm(span=span, adjust=False).mean()` |

## Benchmarks

Run `python benchmarks/bench_ops.py` to compare xpandas ops against their pandas
equivalents. Example output (N=10,000, 20 repeats, median time):

```
Op                         pandas (us)  xpandas (us)   speedup
--------------------------------------------------------------
clip                             481.2           5.6    85.91x >>>
to_datetime                     2706.3          16.6   163.02x >>>
rolling_sum                      142.1           9.1    15.66x >>>
breakout_signal                  187.5          10.3    18.26x >>>
pct_change                       207.7          17.3    11.99x >>>
...
--------------------------------------------------------------
Geometric mean speedup:                                  2.23x
Faster in 23/34 ops
```

Key wins: element-wise ops (`clip`, `compare_*`, `fillna`), rolling window ops
(`rolling_sum/mean/std`), fused ops (`breakout_signal`), and datetime conversion
(`to_datetime` — 163x). Groupby ops are slower because xpandas uses sorted
`std::map` keys (for deterministic TorchScript output) vs pandas' optimized
Cython hashmaps.

### Wrapper Benchmarks

Run `python benchmarks/bench_wrappers.py` to measure Python wrapper overhead and
end-to-end Alpha performance. Example output (N=10,000, 30 repeats, median time):

**Part 1: Wrapper Overhead**

| Operation | Direct (μs) | Wrapper (μs) | Overhead |
|-----------|-------------|--------------|---------|
| Series.__gt__ | 9.4 | 9.7 | +4% |
| Series.__lt__ | 9.4 | 9.9 | +5% |
| Series.__sub__ | 3.7 | 3.9 | +7% |
| Series.astype(float) | 9.9 | 10.2 | +3% |
| DataFrame.__getattr__ | 0.1 | 0.7 | +596% |
| GroupBy→OHLC chain | 127.5 | 517.9 | +306% |
| OHLC×4 cached | 509.7 | 131.1 | -74% 🏆 |

**Part 2: End-to-End Alpha (pandas vs xpandas)**

| Size | Instruments | Pandas (μs) | xpandas (μs) | Speedup |
|------|-------------|-------------|--------------|---------|
| Small (10×50) | 10 | 911 | 69 | 13.2× |
| Medium (50×100) | 50 | 1013 | 302 | 3.4× |
| Large (200×500) | 200 | 2787 | 5870 | 0.47× |
| Geomean | — | — | — | 2.76× |

Wrapper overhead on element-wise ops is negligible (<10%). The `GroupBy→OHLC`
chain shows high overhead from Python dispatch, but per-column OHLC caching
(computing all 4 aggregations in one C++ call) reduces total time by 74%.
At medium scale (50 instruments × 100 ticks), xpandas is 3.4× faster than pandas.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) ([中文](docs/CONTRIBUTING_zh.md)) for a
step-by-step guide to adding a new op, using `rank` as a worked example.

## License

Apache-2.0. See [LICENSE](LICENSE).
