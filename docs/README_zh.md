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

- 像使用 `pandas.DataFrame` 一样使用 `xpandas.DataFrame` —— 相同 API，零改写
- 数值列为 `float64` 张量，字符串列通过枚举编码为 `int64` 张量
- 内部实现：每个类 pandas 操作会分发到对应的 `torch.ops.xpandas.*` C++ 算子

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
    groupby_agg.cpp        # groupby_sum/mean/count/std
    groupby_minmax.cpp     # groupby_min/max/first/last
    rolling.cpp            # rolling_sum/mean/std
    rolling_minmax.cpp     # rolling_min/max（O(n) 单调队列）
    shift.cpp              # shift（前移/后移）
    fillna.cpp             # fillna
    where.cpp              # where_, masked_fill
    pct_change.cpp         # pct_change
    cumulative.cpp         # cumsum, cumprod
    clip.cpp               # clip
    math_ops.cpp           # abs_, log_, zscore
    ewm.cpp                # ewm_mean
    sort.cpp               # sort_by
inference/
  main.cpp                 # 纯 C++ 推理驱动程序
examples/
  alpha_original.py        # 基于原始 pandas 的 Alpha（参考实现）
  alpha_ts.py              # TorchScript 兼容的 Alpha（突破策略）
  alpha_vwap.py            # TorchScript VWAP 均值回归 Alpha
  alpha_momentum.py        # TorchScript 动量 z-score Alpha
  trace_and_save.py        # 编译 + 测试 + 保存 alpha.pt
benchmarks/
  bench_ops.py             # xpandas vs pandas 性能对比
tests/
  test_ops.py                  # 各 C++ 算子单元测试（110 个）
  test_wrappers.py             # 包装器 API 测试（233 个）
  test_alpha_e2e.py            # 端到端 TorchScript 测试（10 个）
  test_alpha_xpandas_e2e.py    # 端到端 xpandas 包装器测试（5 个）
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
pip install --no-build-isolation -e .
```

> **注意：** 需要使用 `--no-build-isolation` 以确保 C++ 扩展与已安装的 PyTorch 使用相同的 ABI 编译。

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

## 可用算子（共 35 个）

### DataFrame 工具

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `lookup` | `(Dict(str, Tensor) table, str key) -> Tensor` | `df['col']` |
| `sort_by` | `(Dict(str, Tensor) table, str by, bool ascending) -> Dict(str, Tensor)` | `df.sort_values(by)` |

### 分组 / 聚合

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `groupby_resample_ohlc` | `(Tensor key, Tensor value) -> (Tensor, Tensor, Tensor, Tensor, Tensor)` | `df.groupby(key)[val].resample().{first,max,min,last}()` |
| `groupby_sum` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].sum()` |
| `groupby_mean` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].mean()` |
| `groupby_count` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].count()` |
| `groupby_std` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].std()` |
| `groupby_min` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].min()` |
| `groupby_max` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].max()` |
| `groupby_first` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].first()` |
| `groupby_last` | `(Tensor key, Tensor value) -> (Tensor, Tensor)` | `df.groupby(key)[val].last()` |

### 逐元素比较

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `compare_gt` | `(Tensor a, Tensor b) -> Tensor` | `series > series` |
| `compare_lt` | `(Tensor a, Tensor b) -> Tensor` | `series < series` |

### 类型转换

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `bool_to_float` | `(Tensor x) -> Tensor` | `series.astype(float)` |

### 融合信号

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `breakout_signal` | `(Tensor price, Tensor high, Tensor low) -> Tensor` | `(price > high).float() - (price < low).float()` |

### 统计

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `rank` | `(Tensor x) -> Tensor` | `series.rank(method='average')` |
| `zscore` | `(Tensor x) -> Tensor` | `(series - series.mean()) / series.std()` |

### 日期时间

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `to_datetime` | `(Tensor epochs, str unit) -> Tensor` | `pd.to_datetime(series, unit=...)` |
| `dt_floor` | `(Tensor dt_ns, int interval_ns) -> Tensor` | `series.dt.floor(freq)` |

### 滚动窗口

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `rolling_sum` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).sum()` |
| `rolling_mean` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).mean()` |
| `rolling_std` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).std()` |
| `rolling_min` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).min()` |
| `rolling_max` | `(Tensor x, int window) -> Tensor` | `series.rolling(window).max()` |

### 移位 / 滞后

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `shift` | `(Tensor x, int periods) -> Tensor` | `series.shift(periods)` |

### NaN 处理

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `fillna` | `(Tensor x, float fill_value) -> Tensor` | `series.fillna(value)` |

### 条件选择

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `where_` | `(Tensor cond, Tensor x, Tensor other) -> Tensor` | `series.where(cond, other)` |
| `masked_fill` | `(Tensor x, Tensor mask, float fill_value) -> Tensor` | `series.mask(mask, value)` |

### 百分比变化

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `pct_change` | `(Tensor x, int periods) -> Tensor` | `series.pct_change(periods)` |

### 累积运算

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `cumsum` | `(Tensor x) -> Tensor` | `series.cumsum()` |
| `cumprod` | `(Tensor x) -> Tensor` | `series.cumprod()` |

### 裁剪

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `clip` | `(Tensor x, float lower, float upper) -> Tensor` | `series.clip(lower, upper)` |

### 数学

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `abs_` | `(Tensor x) -> Tensor` | `series.abs()` |
| `log_` | `(Tensor x) -> Tensor` | `np.log(series)` |

### 指数加权

| 算子 | Schema | 对应 pandas 操作 |
|------|--------|-----------------|
| `ewm_mean` | `(Tensor x, int span) -> Tensor` | `series.ewm(span=span, adjust=False).mean()` |

## 性能基准

运行 `python benchmarks/bench_ops.py` 对比 xpandas 算子与 pandas 等价操作的性能。
示例输出（N=10,000，20 次重复，取中位数）：

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
几何平均加速比:                                           2.23x
34 个算子中 23 个更快
```

主要优势：逐元素运算（`clip`、`compare_*`、`fillna`）、滚动窗口运算
（`rolling_sum/mean/std`）、融合运算（`breakout_signal`）及日期时间转换
（`to_datetime` — 163 倍）。分组运算较慢，因为 xpandas 使用有序
`std::map` 键（保证 TorchScript 输出确定性），而 pandas 使用优化的 Cython 哈希表。

### 包装器基准测试

运行 `python benchmarks/bench_wrappers.py` 可测量 Python 包装器开销和端到端 Alpha 性能。示例输出（N=10,000，重复 30 次，取中位数）：

**第一部分：包装器开销**

| 操作 | 直接调用 (μs) | 包装器调用 (μs) | 开销 |
|------|--------------|----------------|------|
| Series.__gt__ | 9.4 | 9.7 | +4% |
| Series.__lt__ | 9.4 | 9.9 | +5% |
| Series.__sub__ | 3.7 | 3.9 | +7% |
| Series.astype(float) | 9.9 | 10.2 | +3% |
| DataFrame.__getattr__ | 0.1 | 0.7 | +596% |
| GroupBy→OHLC 链 | 127.5 | 517.9 | +306% |
| OHLC×4 缓存 | 509.7 | 131.1 | -74% 🏆 |

**第二部分：端到端 Alpha 对比（pandas vs xpandas，滚动均线交叉策略）**

| 规模 | 标的数 | Pandas (ms) | xpandas (ms) | 加速比 |
|------|--------|-------------|--------------|--------|
| 小（10×50）| 10 | 0.3 | 0.0 | 11.9× |
| 中（50×100）| 50 | 0.4 | 0.1 | 7.8× |
| 大（500×10,000）| 500 | 315.4 | 54.2 | 5.8× |

元素级操作的包装器开销可忽略不计（<10%）。xpandas 在所有测试规模下均快于 pandas。在生产规模（500 个标的 × 10,000 tick）下，滚动均线交叉信号计算耗时 **54 ms**，而 pandas 需要 315 ms —— 提速 **5.8 倍**。`GroupBy→OHLC` 链是例外：xpandas 使用有序 `std::map` 键以保证 TorchScript 输出的确定性，对 groupby 密集型工作负载比 pandas 的 Cython 哈希表慢。

## 技术细节

### 内部实现：为什么用 `Dict[str, Tensor]` 而不是自定义类？

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
