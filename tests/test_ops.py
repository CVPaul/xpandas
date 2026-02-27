"""
tests/test_ops.py -- Unit tests for xpandas custom ops.

Run:
    pip install -e .
    pytest tests/ -v
"""

import math
import torch
import xpandas  # noqa: F401 -- loads the custom ops


class TestGroupbyResampleOhlc:
    """Tests for torch.ops.xpandas.groupby_resample_ohlc."""

    def test_basic(self):
        key   = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        value = torch.tensor([100.0, 105.0, 102.0, 200.0, 198.0, 210.0],
                             dtype=torch.double)

        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, value)

        assert inst.tolist() == [0, 1]
        assert o.tolist() == [100.0, 200.0]
        assert h.tolist() == [105.0, 210.0]
        assert l.tolist() == [100.0, 198.0]
        assert c.tolist() == [102.0, 210.0]

    def test_single_group(self):
        key   = torch.tensor([0, 0, 0], dtype=torch.long)
        value = torch.tensor([10.0, 20.0, 15.0], dtype=torch.double)

        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, value)

        assert inst.tolist() == [0]
        assert o.tolist() == [10.0]
        assert h.tolist() == [20.0]
        assert l.tolist() == [10.0]
        assert c.tolist() == [15.0]

    def test_empty(self):
        key   = torch.empty(0, dtype=torch.long)
        value = torch.empty(0, dtype=torch.double)

        inst, o, h, l, c = torch.ops.xpandas.groupby_resample_ohlc(key, value)

        assert inst.numel() == 0


class TestCompare:
    """Tests for torch.ops.xpandas.compare_gt and compare_lt."""

    def test_gt(self):
        a = torch.tensor([5.0, 3.0, 7.0], dtype=torch.double)
        b = torch.tensor([4.0, 4.0, 7.0], dtype=torch.double)
        result = torch.ops.xpandas.compare_gt(a, b)
        assert result.tolist() == [True, False, False]

    def test_lt(self):
        a = torch.tensor([5.0, 3.0, 7.0], dtype=torch.double)
        b = torch.tensor([4.0, 4.0, 7.0], dtype=torch.double)
        result = torch.ops.xpandas.compare_lt(a, b)
        assert result.tolist() == [False, True, False]


class TestBoolToFloat:
    """Tests for torch.ops.xpandas.bool_to_float."""

    def test_basic(self):
        x = torch.tensor([True, False, True])
        result = torch.ops.xpandas.bool_to_float(x)
        assert result.dtype == torch.double
        assert result.tolist() == [1.0, 0.0, 1.0]


class TestLookup:
    """Tests for torch.ops.xpandas.lookup."""

    def test_basic(self):
        table = {
            "price": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
            "volume": torch.tensor([100.0, 200.0, 300.0], dtype=torch.double),
        }
        result = torch.ops.xpandas.lookup(table, "price")
        assert result.tolist() == [1.0, 2.0, 3.0]


class TestBreakoutSignal:
    """Tests for torch.ops.xpandas.breakout_signal."""

    def test_basic(self):
        price = torch.tensor([106.0, 195.0, 103.0], dtype=torch.double)
        high  = torch.tensor([105.0, 210.0, 105.0], dtype=torch.double)
        low   = torch.tensor([100.0, 198.0, 100.0], dtype=torch.double)
        result = torch.ops.xpandas.breakout_signal(price, high, low)
        assert result.tolist() == [1.0, -1.0, 0.0]


class TestRank:
    """Tests for torch.ops.xpandas.rank."""

    def test_basic(self):
        x = torch.tensor([3.0, 1.0, 2.0, 1.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [4.0, 1.5, 3.0, 1.5]

    def test_no_ties(self):
        x = torch.tensor([10.0, 30.0, 20.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [1.0, 3.0, 2.0]

    def test_all_same(self):
        x = torch.tensor([5.0, 5.0, 5.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [2.0, 2.0, 2.0]

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

    def test_single_element(self):
        x = torch.tensor([42.0], dtype=torch.double)
        result = torch.ops.xpandas.rank(x)
        assert result.tolist() == [1.0]
