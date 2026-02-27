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


class TestToDatetime:
    """Tests for torch.ops.xpandas.to_datetime."""

    def test_seconds(self):
        # 2024-01-15 00:00:00 UTC = 1705276800 epoch seconds
        epochs = torch.tensor([1705276800], dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "s")
        assert result.dtype == torch.long
        assert result[0].item() == 1705276800 * 1_000_000_000

    def test_milliseconds(self):
        epochs = torch.tensor([1705276800000], dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "ms")
        assert result[0].item() == 1705276800000 * 1_000_000

    def test_microseconds(self):
        epochs = torch.tensor([1705276800000000], dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "us")
        assert result[0].item() == 1705276800000000 * 1_000

    def test_nanoseconds_passthrough(self):
        ns_val = 1705276800 * 1_000_000_000
        epochs = torch.tensor([ns_val], dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "ns")
        assert result[0].item() == ns_val

    def test_float64_input(self):
        epochs = torch.tensor([1705276800.5], dtype=torch.double)
        result = torch.ops.xpandas.to_datetime(epochs, "s")
        # truncates 0.5 toward zero, then multiplies
        assert result[0].item() == 1705276800 * 1_000_000_000

    def test_multiple_values(self):
        epochs = torch.tensor([1000, 2000, 3000], dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "s")
        expected = [v * 1_000_000_000 for v in [1000, 2000, 3000]]
        assert result.tolist() == expected

    def test_empty(self):
        epochs = torch.empty(0, dtype=torch.long)
        result = torch.ops.xpandas.to_datetime(epochs, "s")
        assert result.numel() == 0


class TestDtFloor:
    """Tests for torch.ops.xpandas.dt_floor."""

    def test_floor_to_day(self):
        # 2024-01-15 10:30:45 UTC = 1705314645 epoch seconds
        ns = 1705314645 * 1_000_000_000
        # 2024-01-15 00:00:00 UTC = 1705276800 epoch seconds
        expected_ns = 1705276800 * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        interval_day = 86400 * 1_000_000_000  # 1D in ns
        result = torch.ops.xpandas.dt_floor(dt, interval_day)
        assert result[0].item() == expected_ns

    def test_floor_to_hour(self):
        # 10:30:45 -> floor to hour -> 10:00:00
        base_day = 1705276800 * 1_000_000_000  # 2024-01-15 00:00:00
        ns = base_day + (10 * 3600 + 30 * 60 + 45) * 1_000_000_000
        expected_ns = base_day + 10 * 3600 * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        interval_hour = 3600 * 1_000_000_000  # 1h in ns
        result = torch.ops.xpandas.dt_floor(dt, interval_hour)
        assert result[0].item() == expected_ns

    def test_floor_to_second(self):
        # 1705314645.123456789 seconds -> floor to second
        ns = 1705314645 * 1_000_000_000 + 123456789
        expected_ns = 1705314645 * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        interval_sec = 1_000_000_000  # 1s in ns
        result = torch.ops.xpandas.dt_floor(dt, interval_sec)
        assert result[0].item() == expected_ns

    def test_already_aligned(self):
        # exact day boundary should be unchanged
        ns = 1705276800 * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        interval_day = 86400 * 1_000_000_000
        result = torch.ops.xpandas.dt_floor(dt, interval_day)
        assert result[0].item() == ns

    def test_multiple_values(self):
        # two timestamps in same day, one in next day
        day1 = 1705276800 * 1_000_000_000   # 2024-01-15
        day2 = 1705363200 * 1_000_000_000   # 2024-01-16
        ts = torch.tensor([
            day1 + 3600 * 1_000_000_000,     # 01:00 on day1
            day1 + 7200 * 1_000_000_000,     # 02:00 on day1
            day2 + 1800 * 1_000_000_000,     # 00:30 on day2
        ], dtype=torch.long)
        interval_day = 86400 * 1_000_000_000
        result = torch.ops.xpandas.dt_floor(ts, interval_day)
        assert result.tolist() == [day1, day1, day2]

    def test_empty(self):
        dt = torch.empty(0, dtype=torch.long)
        result = torch.ops.xpandas.dt_floor(dt, 1_000_000_000)
        assert result.numel() == 0


class TestToDatetimePythonWrapper:
    """Tests for the xpandas.to_datetime / xpandas.dt_floor Python wrappers."""

    def test_to_datetime_wrapper(self):
        import xpandas
        epochs = torch.tensor([1705276800], dtype=torch.long)
        result = xpandas.to_datetime(epochs, unit="s")
        assert result[0].item() == 1705276800 * 1_000_000_000

    def test_dt_floor_wrapper(self):
        import xpandas
        ns = 1705314645 * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        result = xpandas.dt_floor(dt, freq="1D")
        expected_ns = 1705276800 * 1_000_000_000
        assert result[0].item() == expected_ns

    def test_dt_floor_1h(self):
        import xpandas
        base_day = 1705276800 * 1_000_000_000
        ns = base_day + (10 * 3600 + 30 * 60) * 1_000_000_000
        dt = torch.tensor([ns], dtype=torch.long)
        result = xpandas.dt_floor(dt, freq="1h")
        expected = base_day + 10 * 3600 * 1_000_000_000
        assert result[0].item() == expected


# =====================================================================
# New ops tests
# =====================================================================


class TestGroupbySum:
    """Tests for torch.ops.xpandas.groupby_sum."""

    def test_basic(self):
        key = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        val = torch.tensor([1.0, 3.0, 2.0, 4.0, 6.0], dtype=torch.double)
        keys, sums = torch.ops.xpandas.groupby_sum(key, val)
        assert keys.tolist() == [0, 1]
        assert sums.tolist() == [4.0, 12.0]

    def test_single_group(self):
        key = torch.tensor([0, 0, 0], dtype=torch.long)
        val = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        keys, sums = torch.ops.xpandas.groupby_sum(key, val)
        assert keys.tolist() == [0]
        assert sums.tolist() == [6.0]

    def test_empty(self):
        key = torch.empty(0, dtype=torch.long)
        val = torch.empty(0, dtype=torch.double)
        keys, sums = torch.ops.xpandas.groupby_sum(key, val)
        assert keys.numel() == 0


class TestGroupbyMean:
    """Tests for torch.ops.xpandas.groupby_mean."""

    def test_basic(self):
        key = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        val = torch.tensor([1.0, 3.0, 2.0, 4.0], dtype=torch.double)
        keys, means = torch.ops.xpandas.groupby_mean(key, val)
        assert keys.tolist() == [0, 1]
        assert means.tolist() == [2.0, 3.0]

    def test_single_element_groups(self):
        key = torch.tensor([0, 1, 2], dtype=torch.long)
        val = torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        keys, means = torch.ops.xpandas.groupby_mean(key, val)
        assert means.tolist() == [10.0, 20.0, 30.0]


class TestGroupbyCount:
    """Tests for torch.ops.xpandas.groupby_count."""

    def test_basic(self):
        key = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
        keys, counts = torch.ops.xpandas.groupby_count(key, val)
        assert keys.tolist() == [0, 1]
        assert counts.tolist() == [2.0, 3.0]


class TestGroupbyStd:
    """Tests for torch.ops.xpandas.groupby_std."""

    def test_basic(self):
        key = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        val = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], dtype=torch.double)
        keys, stds = torch.ops.xpandas.groupby_std(key, val)
        assert keys.tolist() == [0, 1]
        assert abs(stds[0].item() - 1.0) < 1e-10
        assert abs(stds[1].item() - 10.0) < 1e-10

    def test_single_element_nan(self):
        key = torch.tensor([0], dtype=torch.long)
        val = torch.tensor([5.0], dtype=torch.double)
        keys, stds = torch.ops.xpandas.groupby_std(key, val)
        assert math.isnan(stds[0].item())  # ddof=1, single element -> NaN


class TestRollingSum:
    """Tests for torch.ops.xpandas.rolling_sum."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_sum(x, 3)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())
        assert result[2].item() == 6.0
        assert result[3].item() == 9.0
        assert result[4].item() == 12.0

    def test_window_1(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_sum(x, 1)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.rolling_sum(x, 3)
        assert result.numel() == 0

    def test_window_larger_than_input(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_sum(x, 5)
        assert all(math.isnan(v) for v in result.tolist())


class TestRollingMean:
    """Tests for torch.ops.xpandas.rolling_mean."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_mean(x, 3)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())
        assert result[2].item() == 2.0
        assert result[3].item() == 3.0
        assert result[4].item() == 4.0

    def test_window_1(self):
        x = torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_mean(x, 1)
        assert result.tolist() == [10.0, 20.0, 30.0]


class TestRollingStd:
    """Tests for torch.ops.xpandas.rolling_std."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_std(x, 3)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())
        # std of [1,2,3] = 1.0 (ddof=1)
        assert abs(result[2].item() - 1.0) < 1e-10
        assert abs(result[3].item() - 1.0) < 1e-10
        assert abs(result[4].item() - 1.0) < 1e-10

    def test_window_1_nan(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        result = torch.ops.xpandas.rolling_std(x, 1)
        # std with window=1 and ddof=1 -> NaN
        assert all(math.isnan(v) for v in result.tolist())


class TestShift:
    """Tests for torch.ops.xpandas.shift."""

    def test_shift_forward(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.shift(x, 1)
        assert math.isnan(result[0].item())
        assert result[1].item() == 1.0
        assert result[2].item() == 2.0
        assert result[3].item() == 3.0

    def test_shift_forward_2(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.shift(x, 2)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())
        assert result[2].item() == 1.0
        assert result[3].item() == 2.0

    def test_shift_backward(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.shift(x, -1)
        assert result[0].item() == 2.0
        assert result[1].item() == 3.0
        assert result[2].item() == 4.0
        assert math.isnan(result[3].item())

    def test_shift_zero(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        result = torch.ops.xpandas.shift(x, 0)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.shift(x, 1)
        assert result.numel() == 0


class TestFillna:
    """Tests for torch.ops.xpandas.fillna."""

    def test_basic(self):
        x = torch.tensor([1.0, float('nan'), 3.0, float('nan')], dtype=torch.double)
        result = torch.ops.xpandas.fillna(x, 0.0)
        assert result.tolist() == [1.0, 0.0, 3.0, 0.0]

    def test_no_nans(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        result = torch.ops.xpandas.fillna(x, 0.0)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_all_nans(self):
        x = torch.tensor([float('nan'), float('nan')], dtype=torch.double)
        result = torch.ops.xpandas.fillna(x, -1.0)
        assert result.tolist() == [-1.0, -1.0]

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.fillna(x, 0.0)
        assert result.numel() == 0


class TestWhere:
    """Tests for torch.ops.xpandas.where_."""

    def test_basic(self):
        cond = torch.tensor([True, False, True], dtype=torch.bool)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        other = torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
        result = torch.ops.xpandas.where_(cond, x, other)
        assert result.tolist() == [1.0, 20.0, 3.0]

    def test_all_true(self):
        cond = torch.tensor([True, True], dtype=torch.bool)
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        other = torch.tensor([10.0, 20.0], dtype=torch.double)
        result = torch.ops.xpandas.where_(cond, x, other)
        assert result.tolist() == [1.0, 2.0]


class TestMaskedFill:
    """Tests for torch.ops.xpandas.masked_fill."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        mask = torch.tensor([False, True, False], dtype=torch.bool)
        result = torch.ops.xpandas.masked_fill(x, mask, 0.0)
        assert result.tolist() == [1.0, 0.0, 3.0]

    def test_all_masked(self):
        x = torch.tensor([1.0, 2.0], dtype=torch.double)
        mask = torch.tensor([True, True], dtype=torch.bool)
        result = torch.ops.xpandas.masked_fill(x, mask, -1.0)
        assert result.tolist() == [-1.0, -1.0]


class TestPctChange:
    """Tests for torch.ops.xpandas.pct_change."""

    def test_basic(self):
        x = torch.tensor([100.0, 110.0, 99.0, 115.0], dtype=torch.double)
        result = torch.ops.xpandas.pct_change(x, 1)
        assert math.isnan(result[0].item())
        assert abs(result[1].item() - 0.1) < 1e-10
        assert abs(result[2].item() - (-0.1)) < 1e-10

    def test_periods_2(self):
        x = torch.tensor([100.0, 200.0, 150.0, 300.0], dtype=torch.double)
        result = torch.ops.xpandas.pct_change(x, 2)
        assert math.isnan(result[0].item())
        assert math.isnan(result[1].item())
        assert abs(result[2].item() - 0.5) < 1e-10   # (150-100)/100
        assert abs(result[3].item() - 0.5) < 1e-10   # (300-200)/200

    def test_zero_denominator(self):
        x = torch.tensor([0.0, 1.0], dtype=torch.double)
        result = torch.ops.xpandas.pct_change(x, 1)
        assert math.isnan(result[1].item())  # division by zero

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.pct_change(x, 1)
        assert result.numel() == 0


class TestCumsum:
    """Tests for torch.ops.xpandas.cumsum."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.cumsum(x)
        assert result.tolist() == [1.0, 3.0, 6.0, 10.0]

    def test_single(self):
        x = torch.tensor([5.0], dtype=torch.double)
        result = torch.ops.xpandas.cumsum(x)
        assert result.tolist() == [5.0]

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.cumsum(x)
        assert result.numel() == 0


class TestCumprod:
    """Tests for torch.ops.xpandas.cumprod."""

    def test_basic(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.cumprod(x)
        assert result.tolist() == [1.0, 2.0, 6.0, 24.0]

    def test_with_negative(self):
        x = torch.tensor([2.0, -1.0, 3.0], dtype=torch.double)
        result = torch.ops.xpandas.cumprod(x)
        assert result.tolist() == [2.0, -2.0, -6.0]


class TestClip:
    """Tests for torch.ops.xpandas.clip."""

    def test_basic(self):
        x = torch.tensor([1.0, 5.0, 10.0, -3.0], dtype=torch.double)
        result = torch.ops.xpandas.clip(x, 0.0, 8.0)
        assert result.tolist() == [1.0, 5.0, 8.0, 0.0]

    def test_all_in_range(self):
        x = torch.tensor([2.0, 3.0, 4.0], dtype=torch.double)
        result = torch.ops.xpandas.clip(x, 0.0, 10.0)
        assert result.tolist() == [2.0, 3.0, 4.0]

    def test_empty(self):
        x = torch.empty(0, dtype=torch.double)
        result = torch.ops.xpandas.clip(x, 0.0, 1.0)
        assert result.numel() == 0
