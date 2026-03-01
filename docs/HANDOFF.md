# xpandas — Handoff Document

> **Last updated:** 2026-03-01  
> **Branch / HEAD:** `master` @ `3a0f4e2`  
> **Status:** ✅ All tasks complete. Working tree clean (only `.sisyphus/` untracked).

---

## 1. 项目一句话目标

把量化交易策略中的 `import pandas as pd` **只改一行**换成 `import xpandas as pd`，
策略代码零修改即可：

1. 在 Python 中正常运行（用 PyTorch custom C++ ops 替代 pandas）
2. 被 `torch.jit.script` 编译成 TorchScript
3. 用 `torch::jit::load()` 在纯 C++ 推理引擎中运行（无 Python 运行时）

---

## 2. 关键文件速查

| 文件 | 行数 | 用途 |
|------|------|------|
| `xpandas/__init__.py` | 115 | 包入口：加载 `.so`、导出所有公共 API、`concat()` / `to_datetime()` / `dt_floor()` |
| `xpandas/wrappers.py` | 566 | **全部 pandas 兼容层**：`Series`、`DataFrame`、`Rolling`、`EWM`、`Expanding`、`GroupBy`、`_LocIndexer`、`_iLocIndexer` |
| `xpandas/_C.cpython-310-x86_64-linux-gnu.so` | — | 编译好的 C++ 扩展（**未纳入 git**，移动到新 worktree 时必须手动 `cp`） |
| `xpandas/csrc/` | — | C++ op 源码（**不要修改**，除非要新增 op） |
| `alpha.py` | 38 | 参考策略文件（**不要修改**，它是验收标准） |
| `tests/test_wrappers.py` | 708 | 233 个 wrapper API 测试 |
| `tests/test_alpha_xpandas_e2e.py` | 92 | 5 个端到端 xpandas 测试 |
| `tests/test_ops.py` | — | 110 个 C++ op 单元测试 |
| `tests/test_alpha_e2e.py` | — | 10 个 TorchScript 端到端测试 |
| `benchmarks/bench_wrappers.py` | — | Wrapper 开销 + 端到端 Alpha 性能对比 |
| `benchmarks/bench_ops.py` | — | 每个 C++ op 与 pandas 的性能对比 |

---

## 3. 架构概览

```
Python 用户代码 (alpha.py)
    import xpandas as pd          ← 唯一修改
    |
    ↓
xpandas/__init__.py               ← 加载 .so，导出 DataFrame/Series/concat 等
    |
    ↓
xpandas/wrappers.py               ← pandas 兼容 API（Series/DataFrame 等）
    |                                内部全部用 torch.Tensor，无 pandas 依赖
    ↓
torch.ops.xpandas.*               ← 35 个注册的 C++ 自定义算子
    |
    ↓
xpandas/csrc/ops/*.cpp            ← 纯 C++ 实现（可被 torch::jit::load 调用）
```

**内部数据模型（实现细节，用户无需关心）：**
- `DataFrame._data` = `Dict[str, Tensor]`（每列一个 1-D tensor）
- `Series._data` = 单个 `Tensor`
- 所有 C++ op 要求 `dtype=torch.float64`，用 `float32` 会崩溃

---

## 4. 已实现的 8 类 API

| 类别 | 关键方法 |
|------|---------|
| **结构类** | `.columns`, `.shape`, `len()`, `.dtypes`, `.rename()`, `.drop()`, `.copy()`, `.head()`, `.tail()`, `.assign()` |
| **算术/比较** | `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`, `==`, `!=`（全部返回 `Series`） |
| **统计/数学** | `.abs()`, `.fillna()`, `.shift()`, `.pct_change()`, `.cumsum()`, `.cumprod()`, `.clip()`, `.where()`, `.mask()`, `.mean()`, `.std()`, `.rank()`, `.zscore()` |
| **窗口** | `.rolling(n)`, `.ewm(span=n)`, `.expanding()` → 代理类，支持 `.mean()/.sum()/.std()/.min()/.max()` |
| **排序/索引** | `.sort_values()`, `.reset_index()`, `.set_index()`, `.value_counts()` |
| **loc/iloc** | `df.loc[...]`, `df.iloc[...]`，布尔 mask 过滤 `df[bool_series]` |
| **apply/map/agg** | `.apply()`, `.map()`, `.agg()`, `.transform()`, `.pipe()`, `.applymap()` |
| **合并/转换** | `.merge()`, `.join()`, `.to_numpy()`, `.to_dict()`, `.describe()`, `.info()`, `concat(axis=0/1)` |

---

## 5. 关键约束（必须遵守）

| 约束 | 说明 |
|------|------|
| `alpha.py` 只改一行 | `import xpandas as pd`，策略文件本身**永不修改** |
| `wrappers.py` 零 pandas 依赖 | 不能 `import pandas`，只能用 `torch` |
| 算术运算返回 `Series` | `Series.__sub__` 等全部返回 `Series`，不返回裸 `Tensor` |
| `float64` 张量 | 所有 C++ op 要求 `dtype=torch.float64` |
| 不新增 C++ op | 除非策略明确需要，否则不动 `csrc/` |
| `.so` 手动复制 | 换 worktree 时执行：`cp xpandas/_C*.so <new-worktree>/xpandas/` |

---

## 6. 性能基准（500×10,000，真实生产规模）

| 规模 | 工具数 | pandas (ms) | xpandas (ms) | 加速比 |
|------|--------|-------------|--------------|--------|
| Small  (10×50)       | 10  | 0.3   | 0.0  | 11.9× |
| Medium (50×100)      | 50  | 0.4   | 0.1  |  7.8× |
| Large  (500×10,000)  | 500 | 315.4 | 54.2 |  5.8× |

**注意：** `GroupBy → OHLC` 链比 pandas 慢（xpandas 用有序 `std::map`，pandas 用 Cython hashmap）。其余场景 xpandas 均更快。

---

## 7. 常用命令

```bash
# 安装
pip install --no-build-isolation -e .

# 运行全部测试（233 个 wrapper + 110 个 op + 15 个 e2e）
python -m pytest tests/ -v

# 验证 alpha.py 一行 import 替换能跑通
python - <<'EOF'
import torch
import xpandas as pd
from alpha import Alpha   # alpha.py 里还是 import pandas as pd，下面换掉
import importlib, sys, types

# 把 alpha.py 里的 pandas 换成 xpandas（仅用于测试，不修改文件）
import importlib.util, builtins
_real_import = builtins.__import__
def _mock_import(name, *args, **kwargs):
    if name == 'pandas':
        return sys.modules.setdefault('pandas', pd)
    return _real_import(name, *args, **kwargs)
builtins.__import__ = _mock_import

import importlib.util
spec = importlib.util.spec_from_file_location("alpha", "alpha.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
builtins.__import__ = _real_import

a = mod.Alpha({})
data = pd.DataFrame({
    'InstrumentID': torch.zeros(5, dtype=torch.long),
    'price': torch.tensor([100., 101., 99., 102., 98.], dtype=torch.float64),
})
a.on_bod(0, data)
result = a.forward(0, data)
sig = result.values if hasattr(result, 'values') else result
print("Signal:", sig.tolist())  # 期望: 全 0.0 或 +1.0/-1.0
EOF

# 移动到新 worktree 时复制 .so
cp xpandas/_C.cpython-310-x86_64-linux-gnu.so <目标路径>/xpandas/

# 运行 benchmark（端到端 Alpha）
python benchmarks/bench_wrappers.py

# 运行 benchmark（逐 op 性能）
python benchmarks/bench_ops.py
```

---

## 8. 提取 Series 结果的标准写法

```python
# forward() 返回 Series（不是裸 Tensor），取出底层张量：
sig = model.forward(0, fwd)
sig_t = sig.values if hasattr(sig, 'values') else sig
```

---

## 9. wrappers.py 结构速查（566 行）

| 行范围 | 内容 |
|--------|------|
| 1–35   | `_iLocIndexer`, `_LocIndexer` |
| 37     | `import torch`（唯一 import） |
| 42–47  | `Index` |
| 49–247 | `Series`（算术、统计、窗口、apply/map/agg） |
| 248–475 | `DataFrame`（结构、排序、loc/iloc、apply、merge/join/转换） |
| 476–519 | `GroupBy`, `GroupByColumn`, `Resampler` |
| 521–566 | `Rolling`, `EWM`, `Expanding` |

---

## 10. 已提交的 Git 历史

```
3a0f4e2  docs: update data model, test list, and E2E benchmark to 500x10000 real scale
673185f  feat(wrappers): add full pandas-compatible API for DataFrame/Series + README benchmarks
60f0f7b  feat(benchmarks): add wrapper overhead + E2E Alpha performance benchmark
58a4d1a  feat(xpandas): add pandas-compat wrapper layer enabling alpha.py import swap
f4678df  Add 11 new ops, AlphaMomentum example, and benchmarks
0f86db1  Add GitHub Actions CI/CD and improve packaging config
```

---

## 11. 已知局限 / 潜在后续工作

| 项目 | 说明 |
|------|------|
| GroupBy 性能 | `GroupBy→OHLC` 比 pandas 慢，因为用了有序 `std::map`；如需加速可改为哈希表（需新 C++ op） |
| GPU 支持 | 目前全部 op 只有 CPU 实现；GPU 需在 `csrc/` 中添加 CUDA kernel |
| `astype` 覆盖范围 | 目前只支持 `astype(float)`，其他类型抛 `NotImplementedError` |
| `loc` 索引 | 目前只支持整数/切片，不支持标签索引 |
| `merge` 多键 | 目前只支持单列等值 merge |

---

## 12. 如何在新 session 中继续

1. **阅读本文件**（`docs/HANDOFF.md`）了解全貌
2. **运行测试**确认环境正常：`python -m pytest tests/ -q`（期望 233 passed）
3. **确认 `.so` 存在**：`ls xpandas/_C*.so`
4. 根据新需求，在 `wrappers.py` 中扩展 API，或在 `csrc/` 中新增 C++ op
5. 新增 op 后参考 `CONTRIBUTING.md` 的 `rank` 示例走完 schema → CPU dispatch → FakeTensor → test 流程

---

*本文件由 Atlas orchestrator 在 2026-03-01 会话结束时自动生成。*
