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
pip install -e .
```

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

## Available Ops

| Op | Schema | Pandas Equivalent |
|----|--------|-------------------|
| `groupby_resample_ohlc` | `(Tensor key, Tensor value) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` | `df.groupby(key)[val].resample().{first,max,min,last}()` |
| `compare_gt` | `(Tensor a, Tensor b) -> Tensor` | `series > series` |
| `compare_lt` | `(Tensor a, Tensor b) -> Tensor` | `series < series` |
| `bool_to_float` | `(Tensor x) -> Tensor` | `series.astype(float)` |
| `lookup` | `(Dict(str, Tensor) table, str key) -> Tensor` | `df['col']` |
| `breakout_signal` | `(Tensor price, Tensor high, Tensor low) -> Tensor` | `(price > high).float() - (price < low).float()` |
| `rank` | `(Tensor x) -> Tensor` | `series.rank(method='average')` |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) ([中文](docs/CONTRIBUTING_zh.md)) for a
step-by-step guide to adding a new op, using `rank` as a worked example.

## License

Apache-2.0. See [LICENSE](LICENSE).
