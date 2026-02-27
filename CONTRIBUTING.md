# Contributing to xpandas

**English** | [中文](docs/CONTRIBUTING_zh.md)

This guide walks through adding a new custom op to xpandas, using **`rank`**
as a complete worked example. By the end, your new op will be:

1. Callable in Python via `torch.ops.xpandas.your_op(...)`
2. Compilable with `torch.jit.script`
3. Executable in pure C++ via `torch::jit::load()`

## Overview: The Four Files You Touch

Adding a new op requires changes in four places:

| # | File | What to do |
|---|------|------------|
| 1 | `xpandas/csrc/ops/<op_name>.cpp` | C++ implementation |
| 2 | `xpandas/csrc/ops/ops.h` | Declare the function signature |
| 3 | `xpandas/csrc/ops/register.cpp` | Register schema + CPU dispatch |
| 4 | `xpandas/ops_meta.py` | FakeTensor kernel (for `torch.compile`) |
| 5 | `tests/test_ops.py` | Unit tests |

The build system (`setup.py` / `CMakeLists.txt`) auto-discovers `*.cpp` files
in `xpandas/csrc/ops/`, so you do **not** need to edit build files.

---

## Step 1: C++ Implementation

Create `xpandas/csrc/ops/rank.cpp`:

```cpp
/**
 * rank.cpp -- Per-element rank within a 1-D tensor.
 *
 * Implements pandas-style Series.rank(method='average', na_option='keep').
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xpandas {

at::Tensor rank(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "rank: input must be 1-D, got ", x.dim(), "-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rank: input must be float64 (Double)");

    const int64_t n = x.size(0);
    if (n == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kDouble));
    }

    auto x_a = x.accessor<double, 1>();

    // Build an index array and sort by value (NaN goes to end).
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        double va = x_a[a], vb = x_a[b];
        if (std::isnan(va)) return false;
        if (std::isnan(vb)) return true;
        return va < vb;
    });

    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));
    auto r_a = result.accessor<double, 1>();

    // Walk through sorted indices and assign average ranks to ties.
    int64_t i = 0;
    while (i < n) {
        double val = x_a[idx[i]];

        if (std::isnan(val)) {
            for (; i < n; ++i) {
                r_a[idx[i]] = std::numeric_limits<double>::quiet_NaN();
            }
            break;
        }

        // Find the run of identical values (ties)
        int64_t j = i + 1;
        while (j < n && !std::isnan(x_a[idx[j]]) && x_a[idx[j]] == val) {
            ++j;
        }

        // Average rank: 1-based ranks from (i+1) to j
        double avg_rank = 0.5 * (double(i + 1) + double(j));
        for (int64_t k = i; k < j; ++k) {
            r_a[idx[k]] = avg_rank;
        }

        i = j;
    }

    return result;
}

} // namespace xpandas
```

Key points:
- Include `"ops.h"` for common headers (`torch/library.h`, `ATen/ATen.h`)
- Put the implementation inside `namespace xpandas { ... }`
- Use `TORCH_CHECK` for input validation (these become readable Python errors)
- Use `at::Tensor` as the return type (aliased from `torch::Tensor`)

## Step 2: Declare in Header

Add the declaration to `xpandas/csrc/ops/ops.h`:

```cpp
namespace xpandas {

// ... existing declarations ...

// rank.cpp
at::Tensor rank(const at::Tensor& x);

} // namespace xpandas
```

## Step 3: Register in Dispatcher

Edit `xpandas/csrc/ops/register.cpp`:

### 3a. Add the schema definition

In the `TORCH_LIBRARY(xpandas, m)` block:

```cpp
m.def("rank(Tensor x) -> Tensor");
```

The schema string uses a simple DSL:
- `Tensor` for tensor arguments
- `int`, `float`, `bool`, `str` for scalars
- `Dict(str, Tensor)` for dict arguments
- `-> Tensor` or `-> (Tensor, Tensor)` for returns

### 3b. Add the CPU implementation

In the `TORCH_LIBRARY_IMPL(xpandas, CPU, m)` block:

```cpp
m.impl("rank", &xpandas::rank);
```

## Step 4: FakeTensor Kernel (Python)

Add to `xpandas/ops_meta.py`:

```python
@torch.library.register_fake("xpandas::rank")
def rank_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)
```

This tells `torch.compile` / `torch.export` the output shape and dtype
without running the actual computation. It is **not** needed for
`torch.jit.script`, but is required for `torch.compile` compatibility.

Rules for FakeTensor kernels:
- Output shape must be derivable from input shapes
- Use `torch.empty(...)` or `torch.empty_like(...)` -- never fill values
- dtype must match the real implementation

## Step 5: Write Tests

Add to `tests/test_ops.py`:

```python
class TestRank:
    def test_basic(self):
        x = torch.tensor([3.0, 1.0, 2.0, 1.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [4.0, 1.5, 3.0, 1.5]

    def test_no_ties(self):
        x = torch.tensor([10.0, 30.0, 20.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [1.0, 3.0, 2.0]

    def test_with_nan(self):
        x = torch.tensor([3.0, float('nan'), 1.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result[0].item() == 2.0
        assert math.isnan(result[1].item())
        assert result[2].item() == 1.0

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.numel() == 0
```

## Step 6: Build and Test

```bash
# Rebuild the extension
pip install -e .

# Run tests
pytest tests/ -v

# Verify it works in TorchScript
python -c "
import torch, xpandas
x = torch.tensor([3.0, 1.0, 2.0], dtype=torch.double)
print(torch.ops.xpandas.rank(x))  # tensor([3., 1., 2.])
"
```

## Step 7 (Optional): Use in a TorchScript Module

```python
class MyAlpha(torch.nn.Module):
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        prices = data["price"]
        return torch.ops.xpandas.rank(prices)

scripted = torch.jit.script(MyAlpha())
scripted.save("my_alpha.pt")
```

This `.pt` file can now be loaded in pure C++ with `torch::jit::load()`,
as long as `libxpandas_ops.so` is loaded first (via `dlopen` or linking).

---

## Checklist for New Ops

- [ ] C++ implementation in `xpandas/csrc/ops/<name>.cpp`
- [ ] Declaration in `xpandas/csrc/ops/ops.h`
- [ ] Schema in `register.cpp` `TORCH_LIBRARY` block
- [ ] CPU impl in `register.cpp` `TORCH_LIBRARY_IMPL` block
- [ ] FakeTensor kernel in `xpandas/ops_meta.py`
- [ ] Tests in `tests/test_ops.py`
- [ ] `pip install -e . && pytest tests/ -v` passes

## Development Setup

```bash
git clone <repo-url>
cd xpandas
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- C++: follow the existing style (C++17, ATen API, `namespace xpandas`)
- Python: standard Python conventions, type hints for all function signatures
- Keep ops simple and focused -- one pandas operation per custom op
