"""
tests/test_edge_cases.py -- Edge case tests for ALL 35 xpandas C++ ops.

Covers scenarios NOT already tested in test_ops.py:
  - NaN propagation / all-NaN
  - Single-element tensors
  - Large tensors (N=100,000)
  - Empty tensors (where not already covered)
  - Negative / zero values
  - Boundary conditions (window sizes, shift periods, etc.)
  - Non-contiguous tensors
  - TORCH_CHECK error paths (wrong dim, wrong dtype, wrong length, missing key)
"""

import math
import torch
import xpandas  # noqa: F401 -- loads the custom ops
import pytest


# ========================== Rolling Mean ==========================

class TestRollingMeanEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 2)
        # index 0: NaN (not enough window), index 1: NaN (contains NaN)
        assert math.isnan(out[0].item())
        assert math.isnan(out[1].item())

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 1)
        assert out.shape == (1,)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 10)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 1)
        assert out.shape == (0,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 2)
        # index 1: mean(-1, -2) = -1.5
        assert out[1].item() == -1.5

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rolling_mean(x, 2)
        # index 0 is NaN (insufficient window), rest are 0
        assert math.isnan(out[0].item())
        for i in range(1, 5):
            assert out[i].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]  # [0, 2, 4, 6, 8]
        out = torch.ops.xpandas.rolling_mean(x2, 2)
        assert out.shape == x2.shape


# ========================== Rolling Sum ==========================

class TestRollingSumEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 2)
        assert math.isnan(out[0].item())
        assert math.isnan(out[1].item())

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 1)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 10)
        assert out.shape == (100000,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 2)
        assert out[1].item() == -3.0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rolling_sum(x, 2)
        assert math.isnan(out[0].item())
        for i in range(1, 5):
            assert out[i].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.rolling_sum(x2, 2)
        assert out.shape == x2.shape


# ========================== Rolling Std ==========================

class TestRollingStdEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 2)
        assert math.isnan(out[0].item())
        assert math.isnan(out[1].item())

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 1)
        assert out.shape == (1,)

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 10)
        assert out.shape == (100000,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 2)
        # std >= 0 for valid windows
        assert out[2].item() >= 0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rolling_std(x, 2)
        # index 0 is NaN (insufficient window), rest are NaN (ddof=1, std of [0,0])
        # Actually std([0,0]) with ddof=1 = 0.0
        assert math.isnan(out[0].item())
        for i in range(1, 5):
            assert out[i].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.rolling_std(x2, 2)
        assert out.shape == x2.shape


# ========================== Rolling Min ==========================

class TestRollingMinEdgeCases:
    def test_nan_propagation(self):
        """rolling_min skips NaN (takes min of valid values in window)."""
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 2)
        assert math.isnan(out[0].item())  # first window-1 positions are NaN
        # rolling_min([1.0, NaN], window=2) -> min of valid = 1.0
        assert out[1].item() == 1.0

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 1)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 10)
        assert out.shape == (100000,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 2)
        assert out[2].item() == -3.0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rolling_min(x, 2)
        assert math.isnan(out[0].item())
        for i in range(1, 5):
            assert out[i].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.rolling_min(x2, 2)
        assert out.shape == x2.shape


# ========================== Rolling Max ==========================

class TestRollingMaxEdgeCases:
    def test_nan_propagation(self):
        """rolling_max skips NaN (takes max of valid values in window)."""
        x = torch.tensor([1.0, math.nan, 3.0, 4.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 2)
        assert math.isnan(out[0].item())  # first window-1 positions are NaN
        # rolling_max([1.0, NaN], window=2) -> max of valid = 1.0
        assert out[1].item() == 1.0

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 1)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 10)
        assert out.shape == (100000,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 2)
        assert out[2].item() == -2.0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rolling_max(x, 2)
        assert math.isnan(out[0].item())
        for i in range(1, 5):
            assert out[i].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.rolling_max(x2, 2)
        assert out.shape == x2.shape


# ========================== Shift ==========================

class TestShiftEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 1)
        # index 0: NaN (shifted), index 1: 1.0, index 2: NaN (original)
        assert math.isnan(out[0].item())
        assert out[1].item() == 1.0
        assert math.isnan(out[2].item())

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 1)
        assert all(math.isnan(v) for v in out.tolist())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 1)
        assert out.shape == (1,)
        assert math.isnan(out[0].item())

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 10)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 1)
        assert out.shape == (0,)

    def test_negative_shift(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.shift(x, -1)
        assert out[0].item() == 2.0
        assert out[1].item() == 3.0
        assert math.isnan(out[2].item())

    def test_boundary_shift_by_n(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        out = torch.ops.xpandas.shift(x, 2)
        assert all(math.isnan(v) for v in out.tolist())

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.shift(x2, 1)
        assert out.shape == x2.shape


# ========================== FillNA ==========================

class TestFillnaEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, 2.0, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 0.0)
        assert out[0].item() == 0.0
        assert out[1].item() == 2.0
        assert out[2].item() == 0.0

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 1.0)
        assert torch.all(out == 1.0)

    def test_single_element(self):
        x = torch.tensor([math.nan], dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 2.0)
        assert out[0].item() == 2.0

    def test_large_tensor(self):
        x = torch.full((100000,), math.nan, dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 3.0)
        assert torch.all(out == 3.0)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 4.0)
        assert out.shape == (0,)

    def test_negative_fill(self):
        x = torch.tensor([math.nan, -2.0], dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, -1.0)
        assert out[0].item() == -1.0
        assert out[1].item() == -2.0

    def test_zero_fill(self):
        x = torch.tensor([math.nan, 0.0], dtype=torch.double)
        out = torch.ops.xpandas.fillna(x, 0.0)
        assert out[0].item() == 0.0
        assert out[1].item() == 0.0

    def test_noncontiguous_tensor(self):
        x = torch.full((10,), math.nan, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.fillna(x2, 5.0)
        assert torch.all(out == 5.0)


# ========================== Where_ ==========================

class TestWhereEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        cond = torch.tensor([True, True, False], dtype=torch.bool)
        other = torch.tensor([0.0, 0.0, 0.0], dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x, other)
        assert out[0].item() == 1.0
        assert math.isnan(out[1].item())
        assert out[2].item() == 0.0

    def test_all_false(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        cond = torch.tensor([False, False], dtype=torch.bool)
        other = torch.tensor([10.0, 20.0], dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x, other)
        assert out.tolist() == [10.0, 20.0]

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        cond = torch.tensor([True], dtype=torch.bool)
        other = torch.tensor([0.0], dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x, other)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        cond = torch.ones(100000, dtype=torch.bool)
        other = torch.zeros(100000, dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x, other)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        cond = torch.empty(0, dtype=torch.bool)
        other = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x, other)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        cond = torch.ones(5, dtype=torch.bool)
        x2 = x[::2]
        other = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.where_(cond, x2, other)
        assert out.shape == x2.shape


# ========================== Masked Fill ==========================

class TestMaskedFillEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        mask = torch.tensor([False, True, False], dtype=torch.bool)
        out = torch.ops.xpandas.masked_fill(x, mask, 0.0)
        assert out[0].item() == 1.0
        assert out[1].item() == 0.0
        assert out[2].item() == 3.0

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        mask = torch.tensor([True], dtype=torch.bool)
        out = torch.ops.xpandas.masked_fill(x, mask, -1.0)
        assert out[0].item() == -1.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        mask = torch.zeros(100000, dtype=torch.bool)
        out = torch.ops.xpandas.masked_fill(x, mask, 0.0)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        mask = torch.empty(0, dtype=torch.bool)
        out = torch.ops.xpandas.masked_fill(x, mask, 0.0)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        mask = torch.zeros(5, dtype=torch.bool)
        out = torch.ops.xpandas.masked_fill(x2, mask, 0.0)
        assert out.shape == x2.shape


# ========================== Compare GT ==========================

class TestCompareGtEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, 2.0], dtype=torch.double)
        y = torch.tensor([1.0, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.compare_gt(x, y)
        assert out.shape == x.shape

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        y = torch.tensor([41.0], dtype=torch.double)
        out = torch.ops.xpandas.compare_gt(x, y)
        assert out[0].item() is True

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        y = torch.zeros(100000, dtype=torch.double)
        out = torch.ops.xpandas.compare_gt(x, y)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        y = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.compare_gt(x, y)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        y = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        y2 = y[::2]
        out = torch.ops.xpandas.compare_gt(x2, y2)
        assert out.shape == x2.shape

    def test_2d_input_passthrough(self):
        """2-D inputs pass through (meta kernel doesn't enforce 1-D)."""
        x = torch.ones(2, 3, dtype=torch.double)
        y = torch.ones(2, 3, dtype=torch.double)
        out = torch.ops.xpandas.compare_gt(x, y)
        assert out.shape == (2, 3)

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise RuntimeError from broadcast."""
        x = torch.ones(3, dtype=torch.double)
        y = torch.ones(5, dtype=torch.double)
        with pytest.raises(RuntimeError):
            torch.ops.xpandas.compare_gt(x, y)


# ========================== Compare LT ==========================

class TestCompareLtEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, 2.0], dtype=torch.double)
        y = torch.tensor([1.0, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.compare_lt(x, y)
        assert out.shape == x.shape

    def test_single_element(self):
        x = torch.tensor([40.0], dtype=torch.double)
        y = torch.tensor([41.0], dtype=torch.double)
        out = torch.ops.xpandas.compare_lt(x, y)
        assert out[0].item() is True

    def test_large_tensor(self):
        x = torch.zeros(100000, dtype=torch.double)
        y = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.compare_lt(x, y)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        y = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.compare_lt(x, y)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        y = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        y2 = y[::2]
        out = torch.ops.xpandas.compare_lt(x2, y2)
        assert out.shape == x2.shape

    def test_2d_input_passthrough(self):
        """2-D inputs pass through (meta kernel doesn't enforce 1-D)."""
        x = torch.ones(2, 3, dtype=torch.double)
        y = torch.ones(2, 3, dtype=torch.double)
        out = torch.ops.xpandas.compare_lt(x, y)
        assert out.shape == (2, 3)

    def test_length_mismatch_raises(self):
        """Mismatched lengths raise RuntimeError from broadcast."""
        x = torch.ones(3, dtype=torch.double)
        y = torch.ones(5, dtype=torch.double)
        with pytest.raises(RuntimeError):
            torch.ops.xpandas.compare_lt(x, y)


# ========================== Bool to Float ==========================

class TestBoolToFloatEdgeCases:
    def test_single_element(self):
        x = torch.tensor([True])
        out = torch.ops.xpandas.bool_to_float(x)
        assert out[0].item() == 1.0
        assert out.dtype == torch.double

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.bool)
        out = torch.ops.xpandas.bool_to_float(x)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.bool)
        out = torch.ops.xpandas.bool_to_float(x)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.ones(10, dtype=torch.bool)
        x2 = x[::2]
        out = torch.ops.xpandas.bool_to_float(x2)
        assert out.shape == x2.shape

    def test_2d_input_passthrough(self):
        """2-D inputs pass through (meta kernel doesn't enforce 1-D)."""
        x = torch.ones(2, 3, dtype=torch.bool)
        out = torch.ops.xpandas.bool_to_float(x)
        assert out.shape == (2, 3)

    def test_all_false(self):
        x = torch.zeros(5, dtype=torch.bool)
        out = torch.ops.xpandas.bool_to_float(x)
        assert torch.all(out == 0.0)

    def test_mixed(self):
        x = torch.tensor([True, False, True, False], dtype=torch.bool)
        out = torch.ops.xpandas.bool_to_float(x)
        assert out.tolist() == [1.0, 0.0, 1.0, 0.0]


# ========================== Lookup ==========================

class TestLookupEdgeCases:
    def test_single_element(self):
        table = {"col": torch.tensor([10.0], dtype=torch.double)}
        out = torch.ops.xpandas.lookup(table, "col")
        assert out[0].item() == 10.0

    def test_large_tensor(self):
        table = {"col": torch.ones(100000, dtype=torch.double)}
        out = torch.ops.xpandas.lookup(table, "col")
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        table = {"col": torch.empty(0, dtype=torch.double)}
        out = torch.ops.xpandas.lookup(table, "col")
        assert out.shape == (0,)

    def test_missing_key_raises(self):
        """Missing key raises IndexError."""
        table = {"col": torch.tensor([1.0], dtype=torch.double)}
        with pytest.raises(IndexError):
            torch.ops.xpandas.lookup(table, "nonexistent")

    def test_multiple_columns(self):
        table = {
            "a": torch.tensor([1.0, 2.0], dtype=torch.double),
            "b": torch.tensor([3.0, 4.0], dtype=torch.double),
        }
        out_a = torch.ops.xpandas.lookup(table, "a")
        out_b = torch.ops.xpandas.lookup(table, "b")
        assert out_a.tolist() == [1.0, 2.0]
        assert out_b.tolist() == [3.0, 4.0]


# ========================== Breakout Signal ==========================

class TestBreakoutSignalEdgeCases:
    def test_nan_propagation(self):
        price = torch.tensor([math.nan, 200.0], dtype=torch.double)
        high = torch.tensor([100.0, 150.0], dtype=torch.double)
        low = torch.tensor([50.0, 100.0], dtype=torch.double)
        out = torch.ops.xpandas.breakout_signal(price, high, low)
        # NaN > 100.0 is false, NaN < 50.0 is false → 0.0
        assert out[0].item() == 0.0
        assert out[1].item() == 1.0

    def test_single_element(self):
        price = torch.tensor([110.0], dtype=torch.double)
        high = torch.tensor([100.0], dtype=torch.double)
        low = torch.tensor([90.0], dtype=torch.double)
        out = torch.ops.xpandas.breakout_signal(price, high, low)
        assert out[0].item() == 1.0

    def test_large_tensor(self):
        price = torch.ones(100000, dtype=torch.double)
        high = torch.ones(100000, dtype=torch.double) * 2
        low = torch.zeros(100000, dtype=torch.double)
        out = torch.ops.xpandas.breakout_signal(price, high, low)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        price = torch.empty(0, dtype=torch.double)
        high = torch.empty(0, dtype=torch.double)
        low = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.breakout_signal(price, high, low)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        price = torch.arange(10, dtype=torch.double)
        high = torch.arange(10, dtype=torch.double) + 1
        low = torch.arange(10, dtype=torch.double) - 1
        p2 = price[::2]
        h2 = high[::2]
        l2 = low[::2]
        out = torch.ops.xpandas.breakout_signal(p2, h2, l2)
        assert out.shape == p2.shape

    def test_wrong_dtype_raises(self):
        """TORCH_CHECK: all inputs must be float64 (Double)."""
        price = torch.tensor([1.0, 2.0], dtype=torch.float)
        high = torch.tensor([1.0, 2.0], dtype=torch.float)
        low = torch.tensor([1.0, 2.0], dtype=torch.float)
        with pytest.raises(RuntimeError, match="Double"):
            torch.ops.xpandas.breakout_signal(price, high, low)

    def test_length_mismatch_raises(self):
        price = torch.tensor([1.0, 2.0], dtype=torch.double)
        high = torch.tensor([1.0], dtype=torch.double)
        low = torch.tensor([1.0, 2.0], dtype=torch.double)
        with pytest.raises(RuntimeError, match="same length"):
            torch.ops.xpandas.breakout_signal(price, high, low)

    def test_wrong_dim_raises(self):
        price = torch.ones(2, 2, dtype=torch.double)
        high = torch.ones(2, 2, dtype=torch.double)
        low = torch.ones(2, 2, dtype=torch.double)
        with pytest.raises(RuntimeError, match="1-D"):
            torch.ops.xpandas.breakout_signal(price, high, low)


# ========================== Rank ==========================

class TestRankEdgeCases:
    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.rank(x)
        assert out[0].item() == 1.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.rank(x)
        assert out.shape == (100000,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0], dtype=torch.double)
        out = torch.ops.xpandas.rank(x)
        # -2.0 is rank 1, -1.0 is rank 2
        assert out[0].item() == 2.0
        assert out[1].item() == 1.0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.rank(x)
        # All same → average rank = 3.0
        assert torch.all(out == 3.0)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.rank(x2)
        assert out.shape == x2.shape

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.rank(x)
        assert all(math.isnan(v) for v in out.tolist())


# ========================== Abs ==========================

class TestAbsEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, -1.0], dtype=torch.double)
        out = torch.ops.xpandas.abs_(x)
        assert math.isnan(out[0].item())
        assert out[1].item() == 1.0

    def test_single_element(self):
        x = torch.tensor([-42.0], dtype=torch.double)
        out = torch.ops.xpandas.abs_(x)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double) * -1
        out = torch.ops.xpandas.abs_(x)
        assert out.shape == (100000,)
        assert torch.all(out == 1.0)

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.abs_(x)
        assert torch.all(out == 0.0)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double) * -1
        x2 = x[::2]
        out = torch.ops.xpandas.abs_(x2)
        assert out.shape == x2.shape


# ========================== Log ==========================

class TestLogEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, 1.0], dtype=torch.double)
        out = torch.ops.xpandas.log_(x)
        assert math.isnan(out[0].item())
        assert out[1].item() == 0.0

    def test_single_element(self):
        x = torch.tensor([math.e], dtype=torch.double)
        out = torch.ops.xpandas.log_(x)
        assert abs(out[0].item() - 1.0) < 1e-10

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.log_(x)
        assert out.shape == (100000,)

    def test_negative_returns_nan(self):
        x = torch.tensor([-1.0, -100.0], dtype=torch.double)
        out = torch.ops.xpandas.log_(x)
        assert all(math.isnan(v) for v in out.tolist())

    def test_zero_returns_nan(self):
        """log_(0) returns NaN (C++ impl: x <= 0 -> NaN)."""
        x = torch.tensor([0.0], dtype=torch.double)
        out = torch.ops.xpandas.log_(x)
        assert math.isnan(out[0].item())

    def test_noncontiguous_tensor(self):
        x = torch.arange(1, 11, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.log_(x2)
        assert out.shape == x2.shape


# ========================== Zscore ==========================

class TestZscoreEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0, 5.0], dtype=torch.double)
        out = torch.ops.xpandas.zscore(x)
        assert math.isnan(out[1].item())

    def test_all_nan(self):
        x = torch.tensor([math.nan, math.nan], dtype=torch.double)
        out = torch.ops.xpandas.zscore(x)
        assert all(math.isnan(v) for v in out.tolist())

    def test_large_tensor(self):
        x = torch.randn(100000, dtype=torch.double)
        out = torch.ops.xpandas.zscore(x)
        assert out.shape == (100000,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.zscore(x2)
        assert out.shape == x2.shape


# ========================== EWM Mean ==========================

class TestEwmMeanEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 3)
        # EWM behavior with NaN depends on implementation
        assert out.shape == (3,)

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 3)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 10)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 5)
        assert out.shape == (0,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 3)
        assert out[0].item() == -1.0
        assert out[1].item() < 0

    def test_zero_values(self):
        x = torch.zeros(5, dtype=torch.double)
        out = torch.ops.xpandas.ewm_mean(x, 3)
        assert torch.all(out == 0.0)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.ewm_mean(x2, 3)
        assert out.shape == x2.shape


# ========================== Pct Change ==========================

class TestPctChangeEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.pct_change(x, 1)
        assert math.isnan(out[0].item())  # first is always NaN

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.pct_change(x, 1)
        assert out.shape == (1,)
        assert math.isnan(out[0].item())

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.pct_change(x, 1)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.pct_change(x, 1)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(1, 11, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.pct_change(x2, 1)
        assert out.shape == x2.shape

    def test_negative_values(self):
        x = torch.tensor([-2.0, -1.0], dtype=torch.double)
        out = torch.ops.xpandas.pct_change(x, 1)
        # (-1 - (-2)) / (-2) = -0.5
        assert abs(out[1].item() - (-0.5)) < 1e-10


# ========================== Cumsum ==========================

class TestCumsumEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.cumsum(x)
        assert out[0].item() == 1.0
        assert math.isnan(out[1].item())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.cumsum(x)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.cumsum(x)
        assert out[-1].item() == 100000.0

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.cumsum(x)
        assert out.shape == (0,)

    def test_negative_values(self):
        x = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.double)
        out = torch.ops.xpandas.cumsum(x)
        assert out.tolist() == [-1.0, -3.0, -6.0]

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.cumsum(x2)
        assert out.shape == x2.shape


# ========================== Cumprod ==========================

class TestCumprodEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([2.0, math.nan, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out[0].item() == 2.0
        assert math.isnan(out[1].item())

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.ones(100000, dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out[-1].item() == 1.0

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out.shape == (0,)

    def test_zero_in_product(self):
        x = torch.tensor([2.0, 0.0, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out.tolist() == [2.0, 0.0, 0.0]

    def test_negative_values(self):
        x = torch.tensor([2.0, -1.0, 3.0], dtype=torch.double)
        out = torch.ops.xpandas.cumprod(x)
        assert out.tolist() == [2.0, -2.0, -6.0]

    def test_noncontiguous_tensor(self):
        x = torch.arange(1, 11, dtype=torch.double)
        x2 = x[::2]
        out = torch.ops.xpandas.cumprod(x2)
        assert out.shape == x2.shape


# ========================== Clip ==========================

class TestClipEdgeCases:
    def test_nan_propagation(self):
        x = torch.tensor([math.nan, 5.0], dtype=torch.double)
        out = torch.ops.xpandas.clip(x, 0.0, 10.0)
        assert math.isnan(out[0].item())
        assert out[1].item() == 5.0

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        out = torch.ops.xpandas.clip(x, 0.0, 100.0)
        assert out[0].item() == 42.0

    def test_large_tensor(self):
        x = torch.randn(100000, dtype=torch.double) * 100
        out = torch.ops.xpandas.clip(x, -1.0, 1.0)
        assert out.shape == (100000,)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.double)
        out = torch.ops.xpandas.clip(x, 0.0, 1.0)
        assert out.shape == (0,)

    def test_negative_values(self):
        x = torch.tensor([-5.0, -1.0], dtype=torch.double)
        out = torch.ops.xpandas.clip(x, -2.0, 0.0)
        assert out.tolist() == [-2.0, -1.0]

    def test_all_below_lower(self):
        x = torch.tensor([-10.0, -20.0], dtype=torch.double)
        out = torch.ops.xpandas.clip(x, 0.0, 100.0)
        assert torch.all(out == 0.0)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.double) * 10
        x2 = x[::2]
        out = torch.ops.xpandas.clip(x2, 10.0, 50.0)
        assert out.shape == x2.shape


# ========================== To Datetime ==========================

class TestToDatetimeEdgeCases:
    def test_single_element(self):
        x = torch.tensor([1000], dtype=torch.long)
        out = torch.ops.xpandas.to_datetime(x, "s")
        assert out[0].item() == 1000 * 1_000_000_000

    def test_large_tensor(self):
        x = torch.arange(100000, dtype=torch.long)
        out = torch.ops.xpandas.to_datetime(x, "s")
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.long)
        out = torch.ops.xpandas.to_datetime(x, "s")
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.long)
        x2 = x[::2]
        out = torch.ops.xpandas.to_datetime(x2, "s")
        assert out.shape == x2.shape


# ========================== Dt Floor ==========================

class TestDtFloorEdgeCases:
    def test_single_element(self):
        ns = 1705314645 * 1_000_000_000
        x = torch.tensor([ns], dtype=torch.long)
        interval = 86400 * 1_000_000_000  # 1 day
        out = torch.ops.xpandas.dt_floor(x, interval)
        expected = 1705276800 * 1_000_000_000
        assert out[0].item() == expected

    def test_large_tensor(self):
        x = torch.arange(100000, dtype=torch.long) * 1_000_000_000
        out = torch.ops.xpandas.dt_floor(x, 3600 * 1_000_000_000)
        assert out.shape == (100000,)

    def test_empty_tensor(self):
        x = torch.empty(0, dtype=torch.long)
        out = torch.ops.xpandas.dt_floor(x, 1_000_000_000)
        assert out.shape == (0,)

    def test_noncontiguous_tensor(self):
        x = torch.arange(10, dtype=torch.long) * 1_000_000_000
        x2 = x[::2]
        out = torch.ops.xpandas.dt_floor(x2, 1_000_000_000)
        assert out.shape == x2.shape


# ========================== Groupby Sum ==========================

class TestGroupbySumEdgeCases:
    def test_nan_propagation(self):
        key = torch.tensor([0, 0, 1], dtype=torch.long)
        val = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_sum(key, val)
        # sum with NaN → NaN for group 0
        assert math.isnan(v[0].item())
        assert v[1].item() == 3.0

    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_sum(key, val)
        assert k.tolist() == [0]
        assert v.tolist() == [10.0]

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.ones(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_sum(key, val)
        assert v[0].item() == 100000.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_sum(k2, v2)
        assert k.shape[0] == v.shape[0]

    def test_negative_values(self):
        key = torch.tensor([0, 0], dtype=torch.long)
        val = torch.tensor([-1.0, -2.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_sum(key, val)
        assert v[0].item() == -3.0


# ========================== Groupby Mean ==========================

class TestGroupbyMeanEdgeCases:
    def test_nan_propagation(self):
        key = torch.tensor([0, 0, 1], dtype=torch.long)
        val = torch.tensor([1.0, math.nan, 3.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_mean(key, val)
        assert math.isnan(v[0].item())
        assert v[1].item() == 3.0

    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_mean(key, val)
        assert v[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.ones(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_mean(key, val)
        assert v[0].item() == 1.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_mean(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Count ==========================

class TestGroupbyCountEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_count(key, val)
        assert v[0].item() == 1.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.ones(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_count(key, val)
        assert v[0].item() == 100000.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_count(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Std ==========================

class TestGroupbyStdEdgeCases:
    def test_nan_propagation(self):
        key = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        val = torch.tensor([1.0, math.nan, 2.0, 4.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_std(key, val)
        assert math.isnan(v[0].item())

    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_std(key, val)
        # single element with ddof=1 → NaN
        assert math.isnan(v[0].item())

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.ones(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_std(key, val)
        # std of all 1s = 0
        assert v[0].item() == 0.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long) % 2
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_std(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Min ==========================

class TestGroupbyMinEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_min(key, val)
        assert v[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.arange(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_min(key, val)
        assert v[0].item() == 0.0

    def test_negative_values(self):
        key = torch.tensor([0, 0], dtype=torch.long)
        val = torch.tensor([-5.0, -10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_min(key, val)
        assert v[0].item() == -10.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_min(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Max ==========================

class TestGroupbyMaxEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_max(key, val)
        assert v[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.arange(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_max(key, val)
        assert v[0].item() == 99999.0

    def test_negative_values(self):
        key = torch.tensor([0, 0], dtype=torch.long)
        val = torch.tensor([-5.0, -10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_max(key, val)
        assert v[0].item() == -5.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_max(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby First ==========================

class TestGroupbyFirstEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_first(key, val)
        assert v[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.arange(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_first(key, val)
        assert v[0].item() == 0.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_first(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Last ==========================

class TestGroupbyLastEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_last(key, val)
        assert v[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.arange(100000, dtype=torch.double)
        k, v = torch.ops.xpandas.groupby_last(key, val)
        assert v[0].item() == 99999.0

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        k, v = torch.ops.xpandas.groupby_last(k2, v2)
        assert k.shape[0] == v.shape[0]


# ========================== Groupby Resample OHLC ==========================

class TestGroupbyResampleOhlcEdgeCases:
    def test_single_element(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([10.0], dtype=torch.double)
        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, val)
        assert inst.shape[0] == 1
        assert o[0].item() == 10.0

    def test_large_tensor(self):
        key = torch.zeros(100000, dtype=torch.long)
        val = torch.arange(100000, dtype=torch.double)
        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, val)
        assert inst.shape[0] == 1

    def test_noncontiguous_tensor(self):
        key = torch.arange(10, dtype=torch.long)
        val = torch.arange(10, dtype=torch.double)
        k2 = key[::2]
        v2 = val[::2]
        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(k2, v2)
        assert inst.shape[0] == k2.shape[0]

    def test_negative_values(self):
        key = torch.tensor([0, 0], dtype=torch.long)
        val = torch.tensor([-5.0, -10.0], dtype=torch.double)
        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, val)
        assert o[0].item() == -5.0
        assert h[0].item() == -5.0
        assert l[0].item() == -10.0
        assert c[0].item() == -10.0


# ========================== Sort By ==========================

class TestSortByEdgeCases:
    def test_single_element(self):
        table = {"col": torch.tensor([42.0], dtype=torch.double)}
        out = torch.ops.xpandas.sort_by(table, "col", True)
        assert out["col"][0].item() == 42.0

    def test_large_tensor(self):
        table = {"col": torch.randn(100000, dtype=torch.double)}
        out = torch.ops.xpandas.sort_by(table, "col", True)
        # verify sorted
        vals = out["col"]
        assert torch.all(vals[1:] >= vals[:-1])

    def test_empty_tensor(self):
        table = {"col": torch.empty(0, dtype=torch.double)}
        out = torch.ops.xpandas.sort_by(table, "col", True)
        assert out["col"].shape == (0,)

    def test_negative_values(self):
        table = {"col": torch.tensor([-3.0, -1.0, -2.0], dtype=torch.double)}
        out = torch.ops.xpandas.sort_by(table, "col", True)
        assert out["col"].tolist() == [-3.0, -2.0, -1.0]

    def test_descending(self):
        table = {"col": torch.tensor([1.0, 3.0, 2.0], dtype=torch.double)}
        out = torch.ops.xpandas.sort_by(table, "col", False)
        assert out["col"].tolist() == [3.0, 2.0, 1.0]

    def test_multiple_columns_preserved(self):
        table = {
            "key": torch.tensor([3.0, 1.0, 2.0], dtype=torch.double),
            "val": torch.tensor([30.0, 10.0, 20.0], dtype=torch.double),
        }
        out = torch.ops.xpandas.sort_by(table, "key", True)
        assert out["key"].tolist() == [1.0, 2.0, 3.0]
        assert out["val"].tolist() == [10.0, 20.0, 30.0]
