# xpandas

[English](../README.md) | **中文**

**唯一一个支持 TorchScript 编译并可在纯 C++ 推理中执行的 pandas 兼容 DataFrame 库 —— 无需 Python 运行时。**

用熟悉的 `pd.DataFrame` / `pd.Series` 语法编写交易策略，通过 `torch.jit.script` 编译，将 `.pt` 产物部署到 C++ 引擎，以原生速度运行。

## 核心特性

- 🔁 **无缝替换** —— `import xpandas as pd` 替换 `import pandas as pd`，零代码改写
- ⚡ **TorchScript 原生** —— 每个算子都是注册的 `TORCH_LIBRARY` C++ 算子，完全兼容 `torch.jit.script`
- 🚀 **纯 C++ 推理** —— `torch::jit::load("alpha.pt")` + `dlopen(libxpandas_ops.so)`，运行时无需 Python
- 📊 **35 个算子** 覆盖 groupby、rolling、ewm、shift、fillna、rank、zscore、cumulative、datetime 等
- 🏎️ **比 pandas 快 2–163 倍** 的逐元素和滚动窗口运算（详见[性能基准](#性能基准)）

## 目标使用场景

| 场景 | 为什么选 xpandas？ |
|------|-----------------|
| **高频 / 量化交易** | 在 Python pandas 中原型开发 Alpha 信号，零改写部署到亚毫秒级 C++ 引擎 |
| **在线模型服务** | 将特征工程（rolling stats、z-score、pct_change）嵌入 TorchScript 模型，在 C++ 中通过 `torch::jit` 服务 |
| **低延迟推理流水线** | 消除 Python GIL 和解释器开销 —— 整条信号路径运行在编译后的 C++ 中 |
| **边缘 / 嵌入式部署** | 仅需一个 `.pt` 文件 + 共享库 —— 目标机器无需安装 Python |

## xpandas 与其他方案的区别

| | **xpandas** | **pandas** | **Polars** | **Modin** | **cuDF (RAPIDS)** |
|---|---|---|---|---|---|
| **核心目标** | TorchScript 编译 + C++ 推理 | 通用数据分析 | 高性能 DataFrame 引擎 | 用并行化扩展 pandas | GPU 加速 DataFrame |
| **`torch.jit.script` 支持** | ✅ 一等公民 —— 每个算子都是 `TORCH_LIBRARY` 自定义算子 | ❌ | ❌ | ❌ | ❌ |
| **纯 C++ 推理** | ✅ `torch::jit::load()` —— 运行时无需 Python | ❌ 需要 Python | ❌ 需要 Rust 运行时 | ❌ 需要 Python | ❌ 需要 Python + CUDA |
| **部署产物** | 单个 `.pt` 文件 + `.so` | Python 源码 + 环境 | Python/Rust 源码 + 环境 | Python 源码 + 环境 | Python 源码 + 环境 |
| **无 Python GIL** | ✅ 所有算子在 C++ 中执行 | ❌ | ✅ (Rust) | 部分 (Ray) | ✅ (GPU) |
| **API 兼容性** | pandas 子集（35 个算子） | 完整 pandas API | 自有 API（类 SQL） | 完整 pandas API | pandas 子集 |
| **最适合** | 量化信号 → C++ 生产 | 探索性分析 | 大规模数据处理 | 扩展现有 pandas 代码 | GPU 批量处理 |

> **一句话总结：** 其他库优化的是*在 Python 中处理数据有多快*。  
> xpandas 解决的是一个根本不同的问题：**将你的 pandas 逻辑完全移出 Python**，编译成可部署的、无 GIL 的 C++ 产物。

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

## API 参考（Python 包装器）

Python 包装器 API（`import xpandas as pd`）提供了 pandas 兼容的类，底层调度到 C++ 算子。

### 核心类

| 类 | 描述 | 关键方法 |
|---|------|---------|
| `pd.DataFrame` | 基于 Dict 的 DataFrame（`Dict[str, Tensor]`） | `__getitem__`、`__setitem__`、`columns`、`shape`、`dtypes`、`head()`、`tail()`、`drop()`、`rename()`、`sort_values()`、`merge()`、`describe()`、`apply()`、`groupby()` |
| `pd.Series` | 1-D Tensor 包装器 | 算术（`+`,`-`,`*`,`/`,`**`,`%`）、比较（`>`,`<`,`>=`,`<=`,`==`,`!=`）、`abs()`、`log()`、`zscore()`、`rank()`、`fillna()`、`shift()`、`pct_change()`、`cumsum()`、`cumprod()`、`clip()`、`where()`、`mask()`、`rolling()`、`ewm()`、`expanding()`、`mean()`、`std()`、`sum()`、`min()`、`max()` |
| `pd.GroupBy` | 分组入口 | `__getitem__(col)` → `GroupByColumn` |
| `pd.GroupByColumn` | 单列分组聚合 | `sum()`、`mean()`、`count()`、`std()`、`min()`、`max()`、`first()`、`last()`、`resample(freq)` → 返回 `(keys, values)` 元组 |
| `pd.Rolling` | 滚动窗口 | `mean()`、`sum()`、`std()`、`min()`、`max()` |
| `pd.EWM` | 指数加权 | `mean()` |
| `pd.Expanding` | 扩展窗口 | `sum()`、`mean()` |
| `pd.Resampler` | OHLC 重采样 | `first()`、`max()`、`min()`、`last()`（缓存机制 —— 一次 C++ 调用获取全部四个值） |
| `pd.Index` | 索引包装器 | `get_level_values()` |

### 模块级函数

| 函数 | 描述 |
|------|------|
| `pd.concat(items, axis=0)` | 拼接 Series（axis=0）或 DataFrame（axis=1） |
| `pd.to_datetime(tensor, unit='s')` | 将 epoch 时间戳转换为纳秒级 datetime 张量 |
| `pd.dt_floor(tensor, freq='1D')` | 将 datetime 张量向下取整到指定频率 |

### 与 pandas 的重要区别

- **所有张量必须为 `torch.double`（float64）**（值列），`torch.long`（int64）（分组键）
- **GroupBy 返回 `(keys_tensor, values_tensor)` 元组**，而非 pandas 风格的分组 DataFrame
- **DataFrame 内部为 `Dict[str, Tensor]`** —— 列顺序取决于插入顺序
- **`to_datetime` 和 `dt_floor` 是模块级函数**，不是 Series 方法

> 完整的类和方法工作示例请参见 `examples/wrapper_api_tour.py`。

## 迁移指南

从 pandas 迁移到 xpandas 非常简单 —— 大部分代码修改是机械性的。

### 第一步：修改 import

```python
# 修改前
import pandas as pd

# 修改后
import xpandas as pd
import torch
```

### 第二步：使用 torch.tensor 代替 Python 列表

```python
# 修改前（pandas）
df = pd.DataFrame({'price': [100.0, 101.5, 99.8]})

# 修改后（xpandas）
df = pd.DataFrame({'price': torch.tensor([100.0, 101.5, 99.8], dtype=torch.double)})
```

### 第三步：适配 GroupBy 结果

```python
# pandas：返回带有分组索引的 DataFrame/Series
result = df.groupby('sector')['price'].mean()
print(result['tech'])  # 基于索引访问

# xpandas：返回 (keys_tensor, values_tensor) 元组
keys, means = df.groupby('sector')['price'].mean()
print(keys, means)  # tensor([0, 1, 2]), tensor([100.5, 98.3, 105.1])
```

### 第四步：使用模块级 datetime 函数

```python
# pandas
df['date'] = pd.to_datetime(df['epoch'], unit='s')
df['date_floor'] = df['date'].dt.floor('1D')

# xpandas
df['date'] = pd.to_datetime(df['epoch'], unit='s')
df['date_floor'] = pd.dt_floor(df['date'], freq='1D')
```

### 常用模式对照表

| pandas | xpandas | 备注 |
|--------|---------|------|
| `df['col']` | `df['col']` | ✅ 相同 |
| `df.col` | `df.col` | ✅ 相同 |
| `series + series` | `series + series` | ✅ 相同 |
| `series.rolling(5).mean()` | `series.rolling(5).mean()` | ✅ 相同 |
| `series.ewm(span=10).mean()` | `series.ewm(span=10).mean()` | ✅ 相同 |
| `series.fillna(0)` | `series.fillna(0)` | ✅ 相同 |
| `df.sort_values('col')` | `df.sort_values(by='col')` | ✅ 相同 |
| `df.merge(other, on='key')` | `df.merge(other, on='key')` | ✅ 相同 |
| `series.where(cond, -1.0)` | `series.where(cond, tensor)` | ⚠️ `other` 必须是 Tensor |
| `df.groupby('k')['v'].sum()` | `keys, vals = df.groupby('k')['v'].sum()` | ⚠️ 返回元组 |
| `pd.to_datetime(s, unit='s')` | `pd.to_datetime(t, unit='s')` | ✅ 相同（模块级） |
| `s.dt.floor('1D')` | `pd.dt_floor(t, freq='1D')` | ⚠️ 模块级函数 |

> 完整的可运行对照示例请参见 `examples/pandas_migration.py`。

## 常见问题 / FAQ

### Q：出现 `RuntimeError: expected scalar type Double` 怎么办？

所有 xpandas 值列必须为 `torch.double`（float64）。检查张量创建代码：

```python
# 错误
t = torch.tensor([1.0, 2.0, 3.0])            # 默认为 float32！

# 正确
t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
```

### Q：GroupBy 报 `Long` 张量相关错误？

GroupBy 键列必须为 `torch.long`（int64）：

```python
df = pd.DataFrame({
    'group': torch.tensor([1, 2, 1, 2], dtype=torch.long),   # int64 键
    'value': torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.double)
})
keys, sums = df.groupby('group')['value'].sum()
```

### Q：`where()` 或 `mask()` 传标量参数报错？

与 pandas 不同，xpandas 要求 `other` 参数必须是 Tensor，不能是标量：

```python
# 错误
result = series.where(cond, -1.0)

# 正确
result = series.where(cond, torch.full_like(series.values, -1.0))
```

### Q：`torch.jit.script()` 编译模型失败怎么办？

1. 确保所有 DataFrame 列都是 Tensor（不是 Python 列表或 NumPy 数组）
2. GroupBy 键必须为 `torch.long`，值必须为 `torch.double`
3. 使用 `pd.to_datetime()` 和 `pd.dt_floor()` 作为模块级调用，而非方法调用
4. 避免在 `@torch.jit.script` 内使用纯 Python 结构（列表推导、f-string 等）

### Q：如何部署到 C++ 推理？

```bash
# 1. 在 Python 中编译并保存
python examples/trace_and_save.py  # 生成 alpha.pt

# 2. 构建 C++ 推理二进制
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
make -j

# 3. 运行 —— 无需 Python
./alpha_infer ../alpha.pt ./libxpandas_ops.so
```

完整的 C++ 驱动代码请参见 `inference/main.cpp`。

### Q：分组运算比 pandas 慢吗？

是的 —— 这是设计选择。xpandas 分组使用有序 `std::map` 键以保证 TorchScript 中输出顺序的确定性。pandas 使用优化的 Cython 哈希表，速度更快但输出顺序不确定。如果分组性能至关重要，考虑预排序数据或减少分组基数。

### Q：能和 `torch.compile` 一起使用吗？

通过 `ops_meta.py` 中的 FakeTensor 核函数提供了基础支持。但主要编译目标是 `torch.jit.script`。生产部署请使用 TorchScript。

### Q：支持 GPU 吗？

目前所有算子仅支持 CPU。算子通过 PyTorch 调度器分发，因此架构上可以添加 CUDA 核函数，但尚未实现。


## 贡献

参见 [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md) 了解如何添加新算子的完整步骤，
以 `rank` 算子为例进行详细说明。

## 许可证

Apache-2.0。参见 [LICENSE](../LICENSE)。
