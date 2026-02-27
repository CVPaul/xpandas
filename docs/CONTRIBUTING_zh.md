# 贡献指南

[English](../CONTRIBUTING.md) | **中文**

本指南以 **`rank`** 算子为完整示例，逐步说明如何向 xpandas 添加一个新的自定义算子。
完成后，你的新算子将：

1. 可在 Python 中通过 `torch.ops.xpandas.your_op(...)` 调用
2. 可被 `torch.jit.script` 编译
3. 可在纯 C++ 环境中通过 `torch::jit::load()` 执行

## 概览：需要修改的文件

添加一个新算子需要修改以下几个文件：

| # | 文件 | 操作 |
|---|------|------|
| 1 | `xpandas/csrc/ops/<算子名>.cpp` | C++ 实现 |
| 2 | `xpandas/csrc/ops/ops.h` | 声明函数签名 |
| 3 | `xpandas/csrc/ops/register.cpp` | 注册 schema + CPU 分发 |
| 4 | `xpandas/ops_meta.py` | FakeTensor 核函数（用于 `torch.compile`） |
| 5 | `tests/test_ops.py` | 单元测试 |

构建系统（`setup.py` / `CMakeLists.txt`）会自动发现 `xpandas/csrc/ops/` 下的所有
`*.cpp` 文件，因此**不需要**修改构建配置。

---

## 第 1 步：C++ 实现

创建 `xpandas/csrc/ops/rank.cpp`：

```cpp
/**
 * rank.cpp -- 一维张量的逐元素排名。
 *
 * 实现 pandas 风格的 Series.rank(method='average', na_option='keep')。
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xpandas {

at::Tensor rank(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "rank: 输入必须是一维张量，实际为 ", x.dim(), " 维");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rank: 输入必须是 float64 (Double) 类型");

    const int64_t n = x.size(0);
    if (n == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kDouble));
    }

    auto x_a = x.accessor<double, 1>();

    // 构建索引数组，按值排序（NaN 排到末尾）
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

    // 遍历排序后的索引，对并列值分配平均排名
    int64_t i = 0;
    while (i < n) {
        double val = x_a[idx[i]];

        if (std::isnan(val)) {
            for (; i < n; ++i) {
                r_a[idx[i]] = std::numeric_limits<double>::quiet_NaN();
            }
            break;
        }

        // 找出相同值的连续区间（并列）
        int64_t j = i + 1;
        while (j < n && !std::isnan(x_a[idx[j]]) && x_a[idx[j]] == val) {
            ++j;
        }

        // 平均排名：基于 1 的排名，从 (i+1) 到 j
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

关键要点：
- 包含 `"ops.h"` 以获取公共头文件（`torch/library.h`、`ATen/ATen.h`）
- 将实现放在 `namespace xpandas { ... }` 中
- 使用 `TORCH_CHECK` 进行输入验证（这些会变成可读的 Python 错误信息）
- 使用 `at::Tensor` 作为返回类型（`torch::Tensor` 的别名）

## 第 2 步：在头文件中声明

在 `xpandas/csrc/ops/ops.h` 中添加声明：

```cpp
namespace xpandas {

// ... 已有声明 ...

// rank.cpp
at::Tensor rank(const at::Tensor& x);

} // namespace xpandas
```

## 第 3 步：注册到调度器

编辑 `xpandas/csrc/ops/register.cpp`：

### 3a. 添加 schema 定义

在 `TORCH_LIBRARY(xpandas, m)` 块中：

```cpp
m.def("rank(Tensor x) -> Tensor");
```

Schema 字符串使用简单的 DSL：
- `Tensor` 表示张量参数
- `int`、`float`、`bool`、`str` 表示标量
- `Dict(str, Tensor)` 表示字典参数
- `-> Tensor` 或 `-> (Tensor, Tensor)` 表示返回值

### 3b. 添加 CPU 实现

在 `TORCH_LIBRARY_IMPL(xpandas, CPU, m)` 块中：

```cpp
m.impl("rank", &xpandas::rank);
```

> **注意：** 如果你的算子没有 Tensor 类型的位置参数（如 `lookup`），调度器无法推断
> 调度键。这种情况下需要将实现直接传给 `m.def()`，使其注册为 catch-all：
> ```cpp
> m.def("lookup(Dict(str, Tensor) table, str key) -> Tensor",
>       &xpandas::lookup);
> ```

## 第 4 步：FakeTensor 核函数（Python）

在 `xpandas/ops_meta.py` 中添加：

```python
@torch.library.register_fake("xpandas::rank")
def rank_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)
```

这告诉 `torch.compile` / `torch.export` 输出的 shape 和 dtype，而无需实际运行计算。
对 `torch.jit.script` 而言**不是必需的**，但为 `torch.compile` 兼容性提供了支持。

FakeTensor 核函数的规则：
- 输出 shape 必须可从输入 shape 推导
- 使用 `torch.empty(...)` 或 `torch.empty_like(...)` —— 不要填充具体值
- dtype 必须与实际实现一致

## 第 5 步：编写测试

在 `tests/test_ops.py` 中添加：

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

## 第 6 步：构建并测试

```bash
# 重新构建扩展
pip install -e .

# 运行测试
pytest tests/ -v

# 验证在 TorchScript 中可用
python -c "
import torch, xpandas
x = torch.tensor([3.0, 1.0, 2.0], dtype=torch.double)
print(torch.ops.xpandas.rank(x))  # tensor([3., 1., 2.])
"
```

## 第 7 步（可选）：在 TorchScript 模块中使用

```python
class MyAlpha(torch.nn.Module):
    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        prices = data["price"]
        return torch.ops.xpandas.rank(prices)

scripted = torch.jit.script(MyAlpha())
scripted.save("my_alpha.pt")
```

这个 `.pt` 文件现在可以在纯 C++ 中通过 `torch::jit::load()` 加载，
只要先加载 `libxpandas_ops.so`（通过 `dlopen` 或链接）。

---

## 新算子检查清单

- [ ] C++ 实现：`xpandas/csrc/ops/<名称>.cpp`
- [ ] 头文件声明：`xpandas/csrc/ops/ops.h`
- [ ] Schema 注册：`register.cpp` 的 `TORCH_LIBRARY` 块
- [ ] CPU 实现注册：`register.cpp` 的 `TORCH_LIBRARY_IMPL` 块
- [ ] FakeTensor 核函数：`xpandas/ops_meta.py`
- [ ] 单元测试：`tests/test_ops.py`
- [ ] `pip install -e . && pytest tests/ -v` 全部通过

## 开发环境搭建

```bash
git clone <仓库地址>
cd xpandas
pip install -e ".[dev]"
pytest tests/ -v
```

## 代码风格

- C++：遵循现有风格（C++17、ATen API、`namespace xpandas`）
- Python：标准 Python 规范，所有函数签名使用类型注解
- 保持算子简洁专注 —— 每个自定义算子对应一个 pandas 操作
