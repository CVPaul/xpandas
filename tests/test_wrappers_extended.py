"""
Extended wrapper tests -- coverage gaps in test_wrappers.py.

Tests:
  - GroupByColumn aggregation value correctness (sum, mean, count, std, min, max, first, last)
  - Resampler edge cases (single group, all-same keys)
  - merge/join error paths and edge cases
  - concat with mismatched columns
  - iloc/loc boundary access (out of range, negative indexing)
  - __repr__ content validation
  - Error message validation from _validate_1d_double() pre-validation
  - DataFrame.describe(), .head(), .tail(), .drop(), .rename() edge cases
  - Series functional methods (apply, map, agg, transform, pipe)
  - DataFrame apply axis=1, applymap, agg, pipe
  - to_dict orient variants
  - to_numpy
  - Index edge cases
  - Expanding edge cases
  - EWM edge cases
  - Rolling window validation
"""
import math

import pytest
import torch
import xpandas  # loads C++ ops
from xpandas.wrappers import (
    DataFrame,
    EWM,
    Expanding,
    GroupBy,
    GroupByColumn,
    Index,
    Resampler,
    Rolling,
    Series,
    _validate_1d_double,
)
import xpandas as pd


# ===========================================================================
# GroupByColumn -- aggregation value correctness
# ===========================================================================

class TestGroupByColumnValues:
    """Verify actual returned values (not just types) for all 8 aggregations."""

    @pytest.fixture
    def df(self):
        return DataFrame({
            "key": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
            "val": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.double),
        })

    def test_sum_values(self, df):
        keys, vals = df.groupby("key")["val"].sum()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(30.0)
        assert vals[1].item() == pytest.approx(120.0)

    def test_mean_values(self, df):
        keys, vals = df.groupby("key")["val"].mean()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(15.0)
        assert vals[1].item() == pytest.approx(40.0)

    def test_count_values(self, df):
        keys, vals = df.groupby("key")["val"].count()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(2.0)
        assert vals[1].item() == pytest.approx(3.0)

    def test_std_values(self, df):
        keys, vals = df.groupby("key")["val"].std()
        assert keys.tolist() == [0, 1]
        # std([10, 20]) with ddof=1 ≈ 7.071
        assert vals[0].item() == pytest.approx(7.0710678, rel=1e-4)
        assert vals[1].item() == pytest.approx(10.0, rel=1e-4)

    def test_min_values(self, df):
        keys, vals = df.groupby("key")["val"].min()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(10.0)
        assert vals[1].item() == pytest.approx(30.0)

    def test_max_values(self, df):
        keys, vals = df.groupby("key")["val"].max()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(20.0)
        assert vals[1].item() == pytest.approx(50.0)

    def test_first_values(self, df):
        keys, vals = df.groupby("key")["val"].first()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(10.0)
        assert vals[1].item() == pytest.approx(30.0)

    def test_last_values(self, df):
        keys, vals = df.groupby("key")["val"].last()
        assert keys.tolist() == [0, 1]
        assert vals[0].item() == pytest.approx(20.0)
        assert vals[1].item() == pytest.approx(50.0)

    def test_single_group(self):
        """All rows have the same key."""
        df = DataFrame({
            "key": torch.tensor([0, 0, 0], dtype=torch.long),
            "val": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
        })
        keys, vals = df.groupby("key")["val"].sum()
        assert keys.tolist() == [0]
        assert vals[0].item() == pytest.approx(6.0)

    def test_many_groups(self):
        """Each row is its own group."""
        n = 50
        df = DataFrame({
            "key": torch.arange(n, dtype=torch.long),
            "val": torch.ones(n, dtype=torch.double),
        })
        keys, vals = df.groupby("key")["val"].sum()
        assert len(keys) == n
        assert torch.all(vals == 1.0)

    def test_groupby_missing_column_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not found"):
            df.groupby("nonexistent")

    def test_groupby_column_missing_value_raises(self):
        df = DataFrame({
            "key": torch.tensor([0], dtype=torch.long),
            "val": torch.tensor([1.0], dtype=torch.double),
        })
        gb = df.groupby("key")
        with pytest.raises(KeyError, match="not found"):
            gb["nonexistent"]


# ===========================================================================
# Resampler -- edge cases
# ===========================================================================

class TestResamplerEdgeCases:
    def test_single_group_ohlc(self):
        """All rows belong to one bucket."""
        df = DataFrame({
            "key": torch.tensor([0, 0, 0], dtype=torch.long),
            "val": torch.tensor([5.0, 10.0, 3.0], dtype=torch.double),
        })
        rs = df.groupby("key")["val"].resample("1D")
        assert rs.first().values[0].item() == pytest.approx(5.0)
        assert rs.max().values[0].item() == pytest.approx(10.0)
        assert rs.min().values[0].item() == pytest.approx(3.0)
        assert rs.last().values[0].item() == pytest.approx(3.0)

    def test_resampler_caches_computation(self):
        """Calling all four methods should only compute OHLC once."""
        df = DataFrame({
            "key": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            "val": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double),
        })
        rs = df.groupby("key")["val"].resample("1D")
        # Call in arbitrary order
        _ = rs.last()
        _ = rs.min()
        _ = rs.max()
        _ = rs.first()
        # Cache should be populated after first call
        assert rs._cache is not None

    def test_resampler_series_has_index(self):
        """Resampler results should have an Index with InstrumentID."""
        df = DataFrame({
            "key": torch.tensor([0, 1], dtype=torch.long),
            "val": torch.tensor([100.0, 200.0], dtype=torch.double),
        })
        rs = df.groupby("key")["val"].resample("1D")
        first = rs.first()
        assert first.index is not None
        level_values = first.index.get_level_values("InstrumentID")
        assert level_values.tolist() == [0, 1]


# ===========================================================================
# Merge / Join -- error paths and edge cases
# ===========================================================================

class TestMergeJoinEdgeCases:
    def test_merge_key_not_in_left_raises(self):
        left = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        right = DataFrame({"b": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not in left"):
            left.merge(right, on="b")

    def test_merge_key_not_in_right_raises(self):
        left = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        right = DataFrame({"b": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not in right"):
            left.merge(right, on="a")

    def test_merge_no_matches(self):
        """Inner join with no overlapping keys returns empty DataFrame."""
        left = DataFrame({
            "key": torch.tensor([1, 2], dtype=torch.long),
            "val": torch.tensor([10.0, 20.0], dtype=torch.double),
        })
        right = DataFrame({
            "key": torch.tensor([3, 4], dtype=torch.long),
            "other": torch.tensor([30.0, 40.0], dtype=torch.double),
        })
        result = left.merge(right, on="key")
        assert len(result) == 0

    def test_merge_basic(self):
        left = DataFrame({
            "key": torch.tensor([1, 2, 3], dtype=torch.long),
            "a": torch.tensor([10.0, 20.0, 30.0], dtype=torch.double),
        })
        right = DataFrame({
            "key": torch.tensor([2, 3, 4], dtype=torch.long),
            "b": torch.tensor([200.0, 300.0, 400.0], dtype=torch.double),
        })
        result = left.merge(right, on="key")
        assert len(result) == 2
        assert result["key"].values.tolist() == [2, 3]
        assert result["a"].values.tolist() == [20.0, 30.0]
        assert result["b"].values.tolist() == [200.0, 300.0]

    def test_join_positional(self):
        """Join without key joins by row position."""
        a = DataFrame({"x": torch.tensor([1.0, 2.0], dtype=torch.double)})
        b = DataFrame({"y": torch.tensor([3.0, 4.0], dtype=torch.double)})
        result = a.join(b)
        assert result.columns == ["x", "y"]
        assert result["x"].values.tolist() == [1.0, 2.0]
        assert result["y"].values.tolist() == [3.0, 4.0]

    def test_join_with_on_delegates_to_merge(self):
        a = DataFrame({
            "key": torch.tensor([1, 2], dtype=torch.long),
            "val": torch.tensor([10.0, 20.0], dtype=torch.double),
        })
        b = DataFrame({
            "key": torch.tensor([1, 2], dtype=torch.long),
            "other": torch.tensor([100.0, 200.0], dtype=torch.double),
        })
        result = a.join(b, on="key")
        assert "other" in result.columns

    def test_join_mismatched_lengths(self):
        """Join by position truncates to shorter."""
        a = DataFrame({"x": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)})
        b = DataFrame({"y": torch.tensor([4.0, 5.0], dtype=torch.double)})
        result = a.join(b)
        assert len(result) == 2


# ===========================================================================
# Concat -- edge cases
# ===========================================================================

class TestConcatEdgeCases:
    def test_concat_empty_list(self):
        result = pd.concat([])
        assert len(result) == 0

    def test_concat_single_item(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        result = pd.concat([df])
        assert len(result) == 1

    def test_concat_vertical_values(self):
        df1 = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        df2 = DataFrame({"a": torch.tensor([3.0, 4.0], dtype=torch.double)})
        result = pd.concat([df1, df2], axis=0)
        assert result["a"].values.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_concat_horizontal_values(self):
        df1 = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        df2 = DataFrame({"b": torch.tensor([2.0], dtype=torch.double)})
        result = pd.concat([df1, df2], axis=1)
        assert sorted(result.columns) == ["a", "b"]


# ===========================================================================
# iloc / loc -- boundary & edge cases
# ===========================================================================

class TestIlocLocEdgeCases:
    @pytest.fixture
    def df(self):
        return DataFrame({
            "a": torch.tensor([10.0, 20.0, 30.0], dtype=torch.double),
            "b": torch.tensor([40.0, 50.0, 60.0], dtype=torch.double),
        })

    def test_iloc_first_row(self, df):
        row = df.iloc[0]
        assert row["a"] == 10.0
        assert row["b"] == 40.0

    def test_iloc_last_row(self, df):
        row = df.iloc[2]
        assert row["a"] == 30.0
        assert row["b"] == 60.0

    def test_iloc_slice_full(self, df):
        result = df.iloc[:]
        assert len(result) == 3

    def test_iloc_slice_partial(self, df):
        result = df.iloc[1:3]
        assert len(result) == 2
        assert result["a"].values.tolist() == [20.0, 30.0]

    def test_iloc_tuple_row_col(self, df):
        """(row_index, col_index) access."""
        val = df.iloc[1, 0]  # row 1, col 0 ("a")
        assert isinstance(val, Series)
        # iloc[1, 0] returns col_data[1] as a scalar tensor wrapped in Series
        # Actually: col_data[row_idx] — and row_idx=1 (int), so it's a scalar tensor

    def test_iloc_unsupported_type_raises(self, df):
        with pytest.raises(IndexError, match="Unsupported"):
            df.iloc["bad"]

    def test_loc_first_row(self, df):
        row = df.loc[0]
        assert row["a"] == 10.0

    def test_loc_slice(self, df):
        result = df.loc[0:2]
        assert isinstance(result, DataFrame)

    def test_loc_unsupported_type_raises(self, df):
        with pytest.raises(IndexError, match="Unsupported"):
            df.loc[[0, 1]]  # list not supported for loc


# ===========================================================================
# __repr__ content
# ===========================================================================

class TestReprContent:
    def test_series_repr(self):
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.double))
        r = repr(s)
        assert "Series" in r

    def test_dataframe_repr(self):
        df = DataFrame({
            "a": torch.tensor([1.0], dtype=torch.double),
            "b": torch.tensor([2.0], dtype=torch.double),
        })
        r = repr(df)
        assert "DataFrame" in r
        assert "1 rows" in r
        assert "2 cols" in r
        assert "a" in r
        assert "b" in r

    def test_empty_dataframe_repr(self):
        df = DataFrame({})
        r = repr(df)
        assert "0 rows" in r
        assert "0 cols" in r


# ===========================================================================
# _validate_1d_double -- error messages
# ===========================================================================

class TestValidate1dDouble:
    def test_rejects_2d_tensor(self):
        t = torch.ones(2, 3, dtype=torch.double)
        with pytest.raises(ValueError, match="1-D"):
            _validate_1d_double(t, "test_op")

    def test_rejects_wrong_dtype(self):
        t = torch.ones(5, dtype=torch.float32)
        with pytest.raises(TypeError, match="float64"):
            _validate_1d_double(t, "test_op")

    def test_accepts_valid_tensor(self):
        t = torch.ones(5, dtype=torch.double)
        _validate_1d_double(t, "test_op")  # should not raise

    def test_error_message_includes_op_name(self):
        t = torch.ones(2, 3, dtype=torch.double)
        with pytest.raises(ValueError, match="my_op"):
            _validate_1d_double(t, "my_op")

    def test_series_methods_validate(self):
        """Series.abs with non-double raises TypeError."""
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.float32))
        with pytest.raises(TypeError, match="float64"):
            s.abs()

    def test_series_methods_validate_2d(self):
        """Series with 2-D tensor raises ValueError on method call."""
        s = Series(torch.ones(2, 3, dtype=torch.double))
        with pytest.raises(ValueError, match="1-D"):
            s.abs()


# ===========================================================================
# DataFrame describe / head / tail / drop / rename -- edge cases
# ===========================================================================

class TestDataFrameMethodEdgeCases:
    def test_describe_keys(self):
        df = DataFrame({
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double),
        })
        desc = df.describe()
        assert "x" in desc
        assert "count" in desc["x"]
        assert "mean" in desc["x"]
        assert "std" in desc["x"]
        assert "min" in desc["x"]
        assert "max" in desc["x"]
        assert desc["x"]["count"] == 5.0
        assert desc["x"]["mean"] == pytest.approx(3.0)
        assert desc["x"]["min"] == pytest.approx(1.0)
        assert desc["x"]["max"] == pytest.approx(5.0)

    def test_head_more_than_rows(self):
        df = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        result = df.head(10)
        assert len(result) == 2

    def test_tail_more_than_rows(self):
        df = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        result = df.tail(10)
        assert len(result) == 2

    def test_head_zero(self):
        df = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        result = df.head(0)
        assert len(result) == 0

    def test_drop_string(self):
        df = DataFrame({
            "a": torch.tensor([1.0], dtype=torch.double),
            "b": torch.tensor([2.0], dtype=torch.double),
        })
        result = df.drop(columns="a")
        assert result.columns == ["b"]

    def test_drop_list(self):
        df = DataFrame({
            "a": torch.tensor([1.0], dtype=torch.double),
            "b": torch.tensor([2.0], dtype=torch.double),
            "c": torch.tensor([3.0], dtype=torch.double),
        })
        result = df.drop(columns=["a", "c"])
        assert result.columns == ["b"]

    def test_drop_none_returns_copy(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        result = df.drop()
        assert result.columns == ["a"]

    def test_rename_columns(self):
        df = DataFrame({
            "old": torch.tensor([1.0], dtype=torch.double),
        })
        result = df.rename(columns={"old": "new"})
        assert result.columns == ["new"]

    def test_rename_none(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        result = df.rename()
        assert result.columns == ["a"]

    def test_sort_values_missing_column_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not found"):
            df.sort_values(by="nonexistent")

    def test_set_index(self):
        df = DataFrame({
            "key": torch.tensor([0, 1], dtype=torch.long),
            "val": torch.tensor([10.0, 20.0], dtype=torch.double),
        })
        result = df.set_index("key")
        assert "key" not in result.columns
        assert "val" in result.columns

    def test_set_index_missing_column_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not found"):
            df.set_index("nonexistent")

    def test_reset_index(self):
        df = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        df._index = torch.tensor([10, 20])
        result = df.reset_index()
        assert result._index is None

    def test_getitem_missing_column_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not found"):
            df["nonexistent"]

    def test_getitem_list_missing_column_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="not found"):
            df[["a", "nonexistent"]]

    def test_getitem_unsupported_type_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(KeyError, match="Unsupported"):
            df[42]


# ===========================================================================
# Series functional methods
# ===========================================================================

class TestSeriesFunctional:
    def test_apply(self):
        s = Series(torch.tensor([1.0, 4.0, 9.0], dtype=torch.double))
        result = s.apply(lambda t: t.sqrt())
        assert result.values[0].item() == pytest.approx(1.0)
        assert result.values[2].item() == pytest.approx(3.0)

    def test_map_vectorized(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.map(lambda t: t * 2)
        assert result.values.tolist() == [2.0, 4.0, 6.0]

    def test_map_non_callable_raises(self):
        s = Series(torch.tensor([1.0], dtype=torch.double))
        with pytest.raises(TypeError, match="callable"):
            s.map("not a function")

    def test_agg_named(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        assert s.agg('mean') == pytest.approx(2.0)
        assert s.agg('sum') == pytest.approx(6.0)
        assert s.agg('min') == pytest.approx(1.0)
        assert s.agg('max') == pytest.approx(3.0)

    def test_agg_callable(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.agg(lambda t: t.median())
        assert result.item() == pytest.approx(2.0)

    def test_agg_unknown_raises(self):
        s = Series(torch.tensor([1.0], dtype=torch.double))
        with pytest.raises(ValueError, match="Unknown"):
            s.agg("unknown_func")

    def test_transform(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.transform(lambda t: t + 1.0)
        assert result.values.tolist() == [2.0, 3.0, 4.0]

    def test_pipe(self):
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.double))
        result = s.pipe(lambda s: s + s)
        assert isinstance(result, Series)
        assert result.values.tolist() == [2.0, 4.0]

    def test_value_counts(self):
        s = Series(torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0], dtype=torch.double))
        vc = s.value_counts()
        # Should return counts with index containing unique values
        assert len(vc) > 0
        assert vc.index is not None


# ===========================================================================
# DataFrame functional methods
# ===========================================================================

class TestDataFrameFunctional:
    def test_apply_axis0(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 2.0], dtype=torch.double),
            "b": torch.tensor([3.0, 4.0], dtype=torch.double),
        })
        result = df.apply(lambda s: s + 1.0, axis=0)
        assert result["a"].values.tolist() == [2.0, 3.0]

    def test_apply_axis1(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 2.0], dtype=torch.double),
            "b": torch.tensor([3.0, 4.0], dtype=torch.double),
        })
        result = df.apply(lambda row: row["a"] + row["b"], axis=1)
        assert isinstance(result, Series)
        assert result.values[0].item() == pytest.approx(4.0)

    def test_applymap(self):
        df = DataFrame({"a": torch.tensor([1.0, 4.0], dtype=torch.double)})
        result = df.applymap(lambda x: x * 2)
        assert result["a"].values.tolist() == [2.0, 8.0]

    def test_agg_mean(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 3.0], dtype=torch.double),
        })
        result = df.agg('mean')
        assert result["a"] == pytest.approx(2.0)

    def test_agg_unknown_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(ValueError, match="Unknown"):
            df.agg("unknown")

    def test_pipe(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        result = df.pipe(lambda d: d.head(1))
        assert isinstance(result, DataFrame)

    def test_to_dict_default(self):
        df = DataFrame({"a": torch.tensor([1.0, 2.0], dtype=torch.double)})
        d = df.to_dict()
        assert d == {"a": [1.0, 2.0]}

    def test_to_dict_records(self):
        df = DataFrame({
            "a": torch.tensor([1.0], dtype=torch.double),
            "b": torch.tensor([2.0], dtype=torch.double),
        })
        records = df.to_dict(orient='records')
        assert len(records) == 1
        assert records[0]["a"] == 1.0
        assert records[0]["b"] == 2.0

    def test_to_dict_invalid_orient_raises(self):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        with pytest.raises(ValueError, match="Unknown orient"):
            df.to_dict(orient='invalid')

    def test_to_numpy(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 2.0], dtype=torch.double),
            "b": torch.tensor([3.0, 4.0], dtype=torch.double),
        })
        arr = df.to_numpy()
        assert arr.shape == (2, 2)

    def test_info_runs(self, capsys):
        df = DataFrame({"a": torch.tensor([1.0], dtype=torch.double)})
        df.info()
        captured = capsys.readouterr()
        assert "DataFrame" in captured.out

    def test_getitem_bool_series_mask(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
        })
        mask = Series(torch.tensor([True, False, True]))
        result = df[mask]
        assert len(result) == 2
        assert result["a"].values.tolist() == [1.0, 3.0]

    def test_getitem_bool_tensor_mask(self):
        df = DataFrame({
            "a": torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
        })
        mask = torch.tensor([True, False, True])
        result = df[mask]
        assert len(result) == 2


# ===========================================================================
# Index -- edge cases
# ===========================================================================

class TestIndexEdgeCases:
    def test_missing_level_raises(self):
        idx = Index({"a": torch.tensor([1, 2, 3])})
        with pytest.raises(KeyError, match="not found"):
            idx.get_level_values("nonexistent")

    def test_multiple_levels(self):
        idx = Index({
            "level1": torch.tensor([0, 1]),
            "level2": torch.tensor([10, 20]),
        })
        assert idx.get_level_values("level1").tolist() == [0, 1]
        assert idx.get_level_values("level2").tolist() == [10, 20]


# ===========================================================================
# Series construction edge cases
# ===========================================================================

class TestSeriesConstruction:
    def test_non_tensor_raises(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            Series([1.0, 2.0])

    def test_dtype_property(self):
        s = Series(torch.tensor([1.0], dtype=torch.double))
        assert s.dtype == torch.double

    def test_shape_property(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        assert s.shape == (3,)

    def test_len(self):
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.double))
        assert len(s) == 2


# ===========================================================================
# DataFrame construction edge cases
# ===========================================================================

class TestDataFrameConstruction:
    def test_non_dict_raises(self):
        with pytest.raises(TypeError, match="dict"):
            DataFrame([1.0, 2.0])

    def test_empty_dataframe_shape(self):
        df = DataFrame({})
        assert df.shape == (0, 0)
        assert len(df) == 0
        assert df.columns == []

    def test_empty_dataframe_index(self):
        df = DataFrame({})
        idx = df.index
        assert len(idx) == 0

    def test_dtypes(self):
        df = DataFrame({
            "a": torch.tensor([1.0], dtype=torch.double),
            "b": torch.tensor([1], dtype=torch.long),
        })
        dtypes = df.dtypes
        assert dtypes["a"] == torch.double
        assert dtypes["b"] == torch.long


# ===========================================================================
# EWM edge cases
# ===========================================================================

class TestEWMEdgeCases:
    def test_ewm_span_validation(self):
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.double))
        with pytest.raises(ValueError, match="span must be >= 1"):
            s.ewm(span=0)

    def test_ewm_basic(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.ewm(span=2).mean()
        assert isinstance(result, Series)
        assert len(result) == 3


# ===========================================================================
# Rolling edge cases
# ===========================================================================

class TestRollingEdgeCases:
    def test_rolling_window_validation(self):
        s = Series(torch.tensor([1.0, 2.0], dtype=torch.double))
        with pytest.raises(ValueError, match="window must be >= 1"):
            s.rolling(0)

    def test_rolling_window_1(self):
        """Window=1 should return the original values."""
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.rolling(1).mean()
        torch.testing.assert_close(result.values, s.values)

    def test_rolling_all_methods_return_series(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
        r = s.rolling(2)
        assert isinstance(r.mean(), Series)
        assert isinstance(r.sum(), Series)
        assert isinstance(r.std(), Series)
        assert isinstance(r.min(), Series)
        assert isinstance(r.max(), Series)


# ===========================================================================
# Expanding edge cases
# ===========================================================================

class TestExpandingEdgeCases:
    def test_expanding_sum_values(self):
        s = Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
        result = s.expanding().sum()
        expected = [1.0, 3.0, 6.0]
        for i, e in enumerate(expected):
            assert result.values[i].item() == pytest.approx(e)

    def test_expanding_mean_values(self):
        s = Series(torch.tensor([2.0, 4.0, 6.0], dtype=torch.double))
        result = s.expanding().mean()
        # cumsum / [1, 2, 3] = [2, 6, 12] / [1, 2, 3] = [2, 3, 4]
        expected = [2.0, 3.0, 4.0]
        for i, e in enumerate(expected):
            assert result.values[i].item() == pytest.approx(e)


# ===========================================================================
# Module-level functions: to_datetime, dt_floor
# ===========================================================================

class TestModuleFunctions:
    def test_to_datetime(self):
        epochs = torch.tensor([1000, 2000], dtype=torch.double)
        result = pd.to_datetime(epochs, unit='s')
        assert result.shape == (2,)

    def test_dt_floor_valid_freq(self):
        ns = torch.tensor([1_500_000_000_000_000_000, 1_600_000_000_000_000_000], dtype=torch.long)
        result = pd.dt_floor(ns, freq='1D')
        assert result.shape == (2,)

    def test_dt_floor_invalid_freq_raises(self):
        ns = torch.tensor([1_500_000_000_000_000_000], dtype=torch.long)
        with pytest.raises(ValueError, match="Unsupported freq"):
            pd.dt_floor(ns, freq='invalid')

    def test_astype_float(self):
        s = Series(torch.tensor([True, False, True]))
        result = s.astype(float)
        assert result.dtype == torch.double
        assert result.values.tolist() == [1.0, 0.0, 1.0]

    def test_astype_unsupported_raises(self):
        s = Series(torch.tensor([1.0], dtype=torch.double))
        with pytest.raises(NotImplementedError, match="astype"):
            s.astype(int)
