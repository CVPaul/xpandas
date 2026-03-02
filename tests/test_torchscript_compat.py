"""
TorchScript compatibility tests for xpandas C++ ops.

Tests:
  - torch.jit.script each op wrapped in a small function
  - Save/load roundtrip for scripted functions
  - Verify scripted output matches eager output
  - lookup and sort_by are CompositeImplicitAutograd (Dict first arg) --
    may need special handling for TorchScript

NOTE: test_alpha_e2e.py already tests 3 full Alpha models with jit.script.
      This file tests individual op-level scripting, NOT full models.
"""
import math
import tempfile

import pytest
import torch
import xpandas  # loads C++ ops


# ===========================================================================
# Helpers
# ===========================================================================

def _check_scripted_matches_eager(scripted_fn, eager_fn, *args, atol=1e-10, rtol=1e-7):
    """Run both scripted and eager, assert outputs match (NaN == NaN)."""
    eager_out = eager_fn(*args)
    scripted_out = scripted_fn(*args)
    if isinstance(eager_out, tuple):
        assert isinstance(scripted_out, tuple)
        for e, s in zip(eager_out, scripted_out):
            torch.testing.assert_close(s, e, atol=atol, rtol=rtol, equal_nan=True)
    elif isinstance(eager_out, dict):
        assert isinstance(scripted_out, dict)
        for k in eager_out:
            torch.testing.assert_close(scripted_out[k], eager_out[k], atol=atol, rtol=rtol, equal_nan=True)
    else:
        torch.testing.assert_close(scripted_out, eager_out, atol=atol, rtol=rtol, equal_nan=True)


def _save_load_roundtrip(scripted_fn):
    """Save a scripted function to temp file and reload it."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    torch.jit.save(scripted_fn, path)
    return torch.jit.load(path)


# ===========================================================================
# Rolling ops
# ===========================================================================

class TestScriptRollingOps:
    def test_rolling_mean(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_mean(x, w)

        x = torch.rand(100, dtype=torch.double)
        eager = torch.ops.xpandas.rolling_mean(x, 5)
        scripted = fn(x, 5)
        torch.testing.assert_close(scripted, eager, equal_nan=True)

    def test_rolling_sum(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_sum(x, w)

        x = torch.rand(100, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, w: torch.ops.xpandas.rolling_sum(x, w), x, 5)

    def test_rolling_std(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_std(x, w)

        x = torch.rand(100, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, w: torch.ops.xpandas.rolling_std(x, w), x, 5)

    def test_rolling_min(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_min(x, w)

        x = torch.rand(100, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, w: torch.ops.xpandas.rolling_min(x, w), x, 5)

    def test_rolling_max(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_max(x, w)

        x = torch.rand(100, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, w: torch.ops.xpandas.rolling_max(x, w), x, 5)


# ===========================================================================
# Element-wise / math ops
# ===========================================================================

class TestScriptMathOps:
    def test_abs(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.abs_(x)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.abs_(x), x)

    def test_log(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.log_(x)

        x = torch.rand(50, dtype=torch.double) + 0.1  # avoid zero
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.log_(x), x)

    def test_zscore(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.zscore(x)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.zscore(x), x)

    def test_rank(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.rank(x)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.rank(x), x)

    def test_cumsum(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.cumsum(x)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.cumsum(x), x)

    def test_cumprod(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.cumprod(x)

        x = torch.rand(20, dtype=torch.double) + 0.5  # avoid too-small values
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.cumprod(x), x)


# ===========================================================================
# Parameterized ops
# ===========================================================================

class TestScriptParameterizedOps:
    def test_shift(self):
        @torch.jit.script
        def fn(x: torch.Tensor, periods: int) -> torch.Tensor:
            return torch.ops.xpandas.shift(x, periods)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, p: torch.ops.xpandas.shift(x, p), x, 3)

    def test_fillna(self):
        @torch.jit.script
        def fn(x: torch.Tensor, fill_value: float) -> torch.Tensor:
            return torch.ops.xpandas.fillna(x, fill_value)

        x = torch.tensor([1.0, float('nan'), 3.0], dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, v: torch.ops.xpandas.fillna(x, v), x, 0.0)

    def test_pct_change(self):
        @torch.jit.script
        def fn(x: torch.Tensor, periods: int) -> torch.Tensor:
            return torch.ops.xpandas.pct_change(x, periods)

        x = torch.tensor([100.0, 110.0, 121.0], dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, p: torch.ops.xpandas.pct_change(x, p), x, 1)

    def test_clip(self):
        @torch.jit.script
        def fn(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
            return torch.ops.xpandas.clip(x, lo, hi)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, lo, hi: torch.ops.xpandas.clip(x, lo, hi), x, -1.0, 1.0)

    def test_ewm_mean(self):
        @torch.jit.script
        def fn(x: torch.Tensor, span: int) -> torch.Tensor:
            return torch.ops.xpandas.ewm_mean(x, span)

        x = torch.randn(50, dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda x, s: torch.ops.xpandas.ewm_mean(x, s), x, 10)


# ===========================================================================
# Conditional ops
# ===========================================================================

class TestScriptConditionalOps:
    def test_where(self):
        @torch.jit.script
        def fn(cond: torch.Tensor, x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.where_(cond, x, other)

        cond = torch.tensor([True, False, True])
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        other = torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        _check_scripted_matches_eager(
            fn,
            lambda c, x, o: torch.ops.xpandas.where_(c, x, o),
            cond, x, other
        )

    def test_masked_fill(self):
        @torch.jit.script
        def fn(x: torch.Tensor, mask: torch.Tensor, fill: float) -> torch.Tensor:
            return torch.ops.xpandas.masked_fill(x, mask, fill)

        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        mask = torch.tensor([True, False, True])
        _check_scripted_matches_eager(
            fn,
            lambda x, m, f: torch.ops.xpandas.masked_fill(x, m, f),
            x, mask, -1.0
        )


# ===========================================================================
# Comparison ops
# ===========================================================================

class TestScriptComparisonOps:
    def test_compare_gt(self):
        @torch.jit.script
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.compare_gt(a, b)

        a = torch.tensor([1.0, 3.0, 2.0], dtype=torch.double)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda a, b: torch.ops.xpandas.compare_gt(a, b), a, b)

    def test_compare_lt(self):
        @torch.jit.script
        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.compare_lt(a, b)

        a = torch.tensor([1.0, 3.0, 2.0], dtype=torch.double)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda a, b: torch.ops.xpandas.compare_lt(a, b), a, b)


# ===========================================================================
# Type casting
# ===========================================================================

class TestScriptCastOps:
    def test_bool_to_float(self):
        @torch.jit.script
        def fn(x: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.bool_to_float(x)

        x = torch.tensor([True, False, True])
        _check_scripted_matches_eager(fn, lambda x: torch.ops.xpandas.bool_to_float(x), x)


# ===========================================================================
# Fused signals
# ===========================================================================

class TestScriptFusedOps:
    def test_breakout_signal(self):
        @torch.jit.script
        def fn(price: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.breakout_signal(price, high, low)

        price = torch.tensor([100.0, 110.0, 90.0], dtype=torch.double)
        high = torch.tensor([105.0, 105.0, 105.0], dtype=torch.double)
        low = torch.tensor([95.0, 95.0, 95.0], dtype=torch.double)
        _check_scripted_matches_eager(
            fn,
            lambda p, h, l: torch.ops.xpandas.breakout_signal(p, h, l),
            price, high, low
        )


# ===========================================================================
# Groupby ops
# ===========================================================================

class TestScriptGroupbyOps:
    @pytest.fixture
    def kv(self):
        key = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        val = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.double)
        return key, val

    def test_groupby_sum(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_sum(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_sum(k, v), key, val)

    def test_groupby_mean(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_mean(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_mean(k, v), key, val)

    def test_groupby_count(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_count(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_count(k, v), key, val)

    def test_groupby_std(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_std(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_std(k, v), key, val)

    def test_groupby_min(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_min(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_min(k, v), key, val)

    def test_groupby_max(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_max(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_max(k, v), key, val)

    def test_groupby_first(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_first(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_first(k, v), key, val)

    def test_groupby_last(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_last(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_last(k, v), key, val)

    def test_groupby_resample_ohlc(self, kv):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_resample_ohlc(k, v)

        key, val = kv
        _check_scripted_matches_eager(fn, lambda k, v: torch.ops.xpandas.groupby_resample_ohlc(k, v), key, val)


# ===========================================================================
# Datetime ops
# ===========================================================================

class TestScriptDatetimeOps:
    def test_to_datetime(self):
        @torch.jit.script
        def fn(epochs: torch.Tensor, unit: str) -> torch.Tensor:
            return torch.ops.xpandas.to_datetime(epochs, unit)

        epochs = torch.tensor([1000, 2000], dtype=torch.double)
        _check_scripted_matches_eager(fn, lambda e, u: torch.ops.xpandas.to_datetime(e, u), epochs, "s")

    def test_dt_floor(self):
        @torch.jit.script
        def fn(dt_ns: torch.Tensor, interval_ns: int) -> torch.Tensor:
            return torch.ops.xpandas.dt_floor(dt_ns, interval_ns)

        ns_per_day = 86400_000_000_000
        dt = torch.tensor([1_500_000_000_000_000_000, 1_500_050_000_000_000_000], dtype=torch.long)
        _check_scripted_matches_eager(fn, lambda d, i: torch.ops.xpandas.dt_floor(d, i), dt, ns_per_day)


# ===========================================================================
# Dict-based ops (CompositeImplicitAutograd)
# ===========================================================================

class TestScriptDictOps:
    def test_lookup(self):
        @torch.jit.script
        def fn(table: dict[str, torch.Tensor], key: str) -> torch.Tensor:
            return torch.ops.xpandas.lookup(table, key)

        table = {"a": torch.tensor([1.0, 2.0], dtype=torch.double)}
        _check_scripted_matches_eager(fn, lambda t, k: torch.ops.xpandas.lookup(t, k), table, "a")

    def test_sort_by(self):
        @torch.jit.script
        def fn(table: dict[str, torch.Tensor], by: str, asc: bool) -> dict[str, torch.Tensor]:
            return torch.ops.xpandas.sort_by(table, by, asc)

        table = {
            "key": torch.tensor([3.0, 1.0, 2.0], dtype=torch.double),
            "val": torch.tensor([30.0, 10.0, 20.0], dtype=torch.double),
        }
        _check_scripted_matches_eager(fn, lambda t, b, a: torch.ops.xpandas.sort_by(t, b, a), table, "key", True)


# ===========================================================================
# Save / Load roundtrip
# ===========================================================================

class TestSaveLoadRoundtrip:
    def test_rolling_mean_save_load(self):
        @torch.jit.script
        def fn(x: torch.Tensor, w: int) -> torch.Tensor:
            return torch.ops.xpandas.rolling_mean(x, w)

        loaded = _save_load_roundtrip(fn)
        x = torch.rand(50, dtype=torch.double)
        expected = fn(x, 5)
        actual = loaded(x, 5)
        torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_groupby_sum_save_load(self):
        @torch.jit.script
        def fn(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.ops.xpandas.groupby_sum(k, v)

        loaded = _save_load_roundtrip(fn)
        k = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        v = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        expected = fn(k, v)
        actual = loaded(k, v)
        for e, a in zip(expected, actual):
            torch.testing.assert_close(a, e, equal_nan=True)

    def test_lookup_save_load(self):
        @torch.jit.script
        def fn(table: dict[str, torch.Tensor], key: str) -> torch.Tensor:
            return torch.ops.xpandas.lookup(table, key)

        loaded = _save_load_roundtrip(fn)
        table = {"col": torch.tensor([1.0, 2.0], dtype=torch.double)}
        expected = fn(table, "col")
        actual = loaded(table, "col")
        torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_breakout_signal_save_load(self):
        @torch.jit.script
        def fn(price: torch.Tensor, high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
            return torch.ops.xpandas.breakout_signal(price, high, low)

        loaded = _save_load_roundtrip(fn)
        p = torch.tensor([100.0, 110.0, 90.0], dtype=torch.double)
        h = torch.tensor([105.0, 105.0, 105.0], dtype=torch.double)
        l = torch.tensor([95.0, 95.0, 95.0], dtype=torch.double)
        expected = fn(p, h, l)
        actual = loaded(p, h, l)
        torch.testing.assert_close(actual, expected, equal_nan=True)


# ===========================================================================
# Composite scripted module (multiple ops chained)
# ===========================================================================

class TestScriptedModule:
    def test_rolling_crossover_module(self):
        """A small scripted module that computes a rolling mean crossover signal."""

        class RollingCrossover(torch.nn.Module):
            def __init__(self, fast: int = 5, slow: int = 20):
                super().__init__()
                self.fast = fast
                self.slow = slow

            def forward(self, price: torch.Tensor) -> torch.Tensor:
                fast_ma = torch.ops.xpandas.rolling_mean(price, self.fast)
                slow_ma = torch.ops.xpandas.rolling_mean(price, self.slow)
                diff = fast_ma - slow_ma
                return torch.ops.xpandas.zscore(diff)

        model = RollingCrossover(fast=3, slow=10)
        scripted = torch.jit.script(model)

        x = torch.randn(100, dtype=torch.double)
        eager = model(x)
        out = scripted(x)
        torch.testing.assert_close(out, eager, equal_nan=True)

    def test_alpha_signal_module(self):
        """A module that combines fillna, pct_change, clip, and zscore."""

        class AlphaSignal(torch.nn.Module):
            def forward(self, price: torch.Tensor) -> torch.Tensor:
                ret = torch.ops.xpandas.pct_change(price, 1)
                ret = torch.ops.xpandas.fillna(ret, 0.0)
                ret = torch.ops.xpandas.clip(ret, -0.1, 0.1)
                return torch.ops.xpandas.zscore(ret)

        model = AlphaSignal()
        scripted = torch.jit.script(model)

        x = torch.tensor([100.0, 102.0, 98.0, 105.0, 101.0], dtype=torch.double)
        eager = model(x)
        out = scripted(x)
        torch.testing.assert_close(out, eager, equal_nan=True)

    def test_module_save_load(self):
        """Scripted module can be saved and loaded."""

        class SimpleAlpha(torch.nn.Module):
            def forward(self, price: torch.Tensor) -> torch.Tensor:
                return torch.ops.xpandas.rolling_mean(price, 5)

        scripted = torch.jit.script(SimpleAlpha())
        loaded = _save_load_roundtrip(scripted)

        x = torch.randn(50, dtype=torch.double)
        expected = scripted(x)
        actual = loaded(x)
        torch.testing.assert_close(actual, expected, equal_nan=True)
