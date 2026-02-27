# xpandas

[English](../README.md) | **中文**

将类 pandas 的 DataFrame 操作实现为 **PyTorch 自定义算子**（`TORCH_LIBRARY`），
支持 `torch.jit.script` 编译，并可在纯 C++ 环境中通过 `torch::jit::load()` 直接推理。

## 为什么需要 xpandas？

量化交易策略通常使用 Python + pandas 进行原型开发。将策略部署到低延迟 C++ 引擎
传统上需要完全重写。xpandas 填补了这个鸿沟：

1. 将 `import pandas as pd` 替换为 `import xpandas as pd`
2. 使用 `torch.jit.script(model)` 将模块编译为 TorchScript
3. 在 C++ 中加载 `.pt` 文件 —— 无需 Python 运行时

## 架构

```
  Python 端                           C++ 端
  ----------                          ------
  import xpandas                      dlopen(libxpandas_ops.so)
  model = Alpha()                     auto m = torch::jit::load("alpha.pt")
  scripted = torch.jit.script(model)  m.get_method("on_bod")({ts, data})
  scripted.save("alpha.pt")           auto sig = m.forward({ts, data})
```

**数据模型：**

- "DataFrame" 即 `Dict[str, Tensor]`（列名 -> 一维张量）
- 字符串列通过枚举编码为 `int64` 张量
- 数值列为 `float64` 张量
- 每个类 pandas 操作都是一个注册的 `torch.ops.xpandas.*` 算子

## 项目结构

```
xpandas/
  __init__.py              # 包初始化，加载 C++ 扩展
  ops_meta.py              # FakeTensor 核函数（用于 torch.compile）
  csrc/ops/
    ops.h                  # 公共头文件，包含算子声明
    register.cpp           # TORCH_LIBRARY schema 定义 + CPU 分发
    groupby_resample_ohlc.cpp
    compare.cpp
    cast.cpp
    lookup.cpp
    breakout_signal.cpp
    rank.cpp               # 示例算子（参见 CONTRIBUTING_zh.md）
    to_datetime.cpp        # to_datetime + dt_floor
inference/
  main.cpp                 # 纯 C++ 推理驱动程序
examples/
  alpha_original.py        # 基于原始 pandas 的 Alpha（参考实现）
  alpha_ts.py              # TorchScript 兼容的 Alpha
  trace_and_save.py        # 编译 + 测试 + 保存 alpha.pt
tests/
  test_ops.py              # 各算子单元测试
  test_alpha_e2e.py        # 端到端编译测试
docs/
  README_zh.md             # 中文文档（本文件）
  CONTRIBUTING_zh.md       # 中文贡献指南
```

## 快速开始

### 环境要求

- Python >= 3.9
- PyTorch >= 2.0
- 支持 C++17 的 C++ 编译器

### 安装（Python）

```bash
pip install -e .
```

### 运行测试

```bash
pytest tests/ -v
```

### 编译并保存模型

```bash
python examples/trace_and_save.py
# 生成 alpha.pt
```

### 构建并运行 C++ 推理

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
make -j

./alpha_infer ../alpha.pt ./libxpandas_ops.so
# 输出: Signal: [+1.0, -1.0]
```

## 可用算子

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `groupby_resample_ohlc` | `(Tensor key, Tensor value) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` | `df.groupby(key)[val].resample().{first,max,min,last}()` |
| `compare_gt` | `(Tensor a, Tensor b) -> Tensor` | `series > series` |
| `compare_lt` | `(Tensor a, Tensor b) -> Tensor` | `series < series` |
| `bool_to_float` | `(Tensor x) -> Tensor` | `series.astype(float)` |
| `lookup` | `(Dict(str, Tensor) table, str key) -> Tensor` | `df['col']` |
| `breakout_signal` | `(Tensor price, Tensor high, Tensor low) -> Tensor` | `(price > high).float() - (price < low).float()` |
| `rank` | `(Tensor x) -> Tensor` | `series.rank(method='average')` |
| `to_datetime` | `(Tensor epochs, str unit) -> Tensor` | `pd.to_datetime(series, unit=...)` |
| `dt_floor` | `(Tensor dt_ns, int interval_ns) -> Tensor` | `series.dt.floor(freq)` |

## 技术细节

### 为什么选择 `Dict[str, Tensor]` 而不是自定义类？

TorchScript 原生支持 `Dict[str, Tensor]` 作为函数参数和返回值。使用自定义类
（`torch.classes.*`）虽然可行，但会增加 C++/Python 双端的维护成本，且序列化行为更复杂。
`Dict[str, Tensor]` 是最简洁的方案。

### 关于 `lookup` 算子的特殊处理

`lookup` 的函数签名 `(Dict[str, Tensor], str) -> Tensor` 不包含任何 Tensor 位置参数，
因此 PyTorch 调度器无法从参数推断调度键（dispatch key）。解决方案是将 `lookup` 注册为
**catch-all** 实现（CompositeImplicitAutograd），而非仅注册到 CPU 调度键。具体做法是
在 `TORCH_LIBRARY` 块中直接将实现传给 `m.def()`：

```cpp
m.def("lookup(Dict(str, Tensor) table, str key) -> Tensor",
      &xpandas::lookup);
```

### FakeTensor 核函数

`ops_meta.py` 中的 FakeTensor 核函数告诉 `torch.compile` / `torch.export` 每个算子
输出的 shape 和 dtype，而无需实际执行计算。这对 `torch.jit.script` 并非必需，但为
`torch.compile` 兼容性提供了支持。

## 贡献

参见 [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md) 了解如何添加新算子的完整步骤，
以 `rank` 算子为例进行详细说明。

## 许可证

Apache-2.0。参见 [LICENSE](../LICENSE)。
