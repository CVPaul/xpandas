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
    rolling.cpp            # rolling_sum/mean/std
    shift.cpp              # shift (lag/lead)
    fillna.cpp             # fillna
    where.cpp              # where_, masked_fill
    pct_change.cpp         # pct_change
    cumulative.cpp         # cumsum, cumprod
    clip.cpp               # clip
inference/
  main.cpp                 # Pure C++ inference driver
examples/
  alpha_original.py        # Original pandas-based Alpha (reference)
  alpha_ts.py              # TorchScript-compatible Alpha
  trace_and_save.py        # Script + test + save alpha.pt
tests/
  test_ops.py              # Unit tests for each op
  test_alpha_e2e.py        # End-to-end scripting test
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

## Available Ops (24 total)

### DataFrame Utilities

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `lookup` | `(Dict(str, Tensor) table, str key) -> Tensor` | `df['col']` |

### Groupby / Aggregation

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `groupby_resample_ohlc` | `(Tensor key, Tensor value) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` | `df.groupby(key)[val].resample().{first,max,min,last}()` |
| `groupby_sum` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].sum()` |
| `groupby_mean` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].mean()` |
| `groupby_count` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].count()` |
| `groupby_std` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].std()` |

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) ([中文](docs/CONTRIBUTING_zh.md)) for a
step-by-step guide to adding a new op, using `rank` as a worked example.

## License

Apache-2.0. See [LICENSE](LICENSE).
