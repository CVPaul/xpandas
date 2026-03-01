import torch
import pytest
import xpandas  # loads C++ ops
from xpandas.wrappers import Series, Index, DataFrame, GroupBy, GroupByColumn, Resampler

class TestSeries:
    def test_creation_from_tensor(self):
        t = torch.tensor([1., 2., 3.], dtype=torch.double)
        s = Series(t)
        assert s._data is t  # same object

    def test_values_property(self):
        t = torch.tensor([1., 2.], dtype=torch.double)
        s = Series(t)
        assert s.values is t  # same object identity

    def test_gt_operator(self):
        a = Series(torch.tensor([100., 106.], dtype=torch.double))
        b = Series(torch.tensor([105., 105.], dtype=torch.double))
        result = a > b
        assert isinstance(result, Series)
        assert result.values.dtype == torch.bool
        assert result.values.tolist() == [False, True]

    def test_lt_operator(self):
        a = Series(torch.tensor([100., 106.], dtype=torch.double))
        b = Series(torch.tensor([105., 105.], dtype=torch.double))
        result = a < b
        assert isinstance(result, Series)
        assert result.values.dtype == torch.bool
        assert result.values.tolist() == [True, False]

    def test_sub_operator_returns_series(self):
        """__sub__ must now return Series (changed from raw Tensor in Phase 3)"""
        a = Series(torch.tensor([3., 1.], dtype=torch.double))
        b = Series(torch.tensor([1., 0.], dtype=torch.double))
        result = a - b
        assert isinstance(result, Series), f"Expected Series but got {type(result)}"
        assert result.values.tolist() == [2., 1.]
    
    def test_sub_returns_series(self):
        """__sub__ must return Series, not raw Tensor"""
        a = Series(torch.tensor([3., 4.], dtype=torch.double))
        b = Series(torch.tensor([1., 2.], dtype=torch.double))
        result = a - b
        assert isinstance(result, Series), f"Expected Series, got {type(result)}"
        assert result.values.tolist() == [2., 2.]

    def test_add_returns_series(self):
        a = Series(torch.tensor([1., 2.], dtype=torch.double))
        b = Series(torch.tensor([3., 4.], dtype=torch.double))
        result = a + b
        assert isinstance(result, Series)
        assert result.values.tolist() == [4., 6.]

    def test_mul_returns_series(self):
        a = Series(torch.tensor([2., 3.], dtype=torch.double))
        b = Series(torch.tensor([4., 5.], dtype=torch.double))
        result = a * b
        assert isinstance(result, Series)
        assert result.values.tolist() == [8., 15.]

    def test_truediv_returns_series(self):
        a = Series(torch.tensor([6., 9.], dtype=torch.double))
        b = Series(torch.tensor([2., 3.], dtype=torch.double))
        result = a / b
        assert isinstance(result, Series)
        assert result.values.tolist() == [3., 3.]

    def test_neg_returns_series(self):
        a = Series(torch.tensor([1., -2.], dtype=torch.double))
        result = -a
        assert isinstance(result, Series)
        assert result.values.tolist() == [-1., 2.]

    def test_arithmetic_with_scalar(self):
        a = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert isinstance(a + 1.0, Series)
        assert isinstance(a * 2.0, Series)
        assert isinstance(a - 0.5, Series)
        assert isinstance(a / 2.0, Series)

    def test_radd_rsub_rmul(self):
        a = Series(torch.tensor([1., 2.], dtype=torch.double))
        assert isinstance(1.0 + a, Series)
        assert isinstance(1.0 - a, Series)
        assert isinstance(2.0 * a, Series)

    def test_comparison_chain_bitwise(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        mask = (s > 1) & (s < 3)
        assert isinstance(mask, Series)
        assert mask.values.tolist() == [False, True, False]

    def test_eq_ne_ge_le(self):
        a = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert isinstance(a == 2.0, Series)
        assert isinstance(a != 2.0, Series)
        assert isinstance(a >= 2.0, Series)
        assert isinstance(a <= 2.0, Series)

    def test_pow_mod(self):
        a = Series(torch.tensor([2., 3.], dtype=torch.double))
        assert isinstance(a ** 2, Series)
        assert isinstance(a % 2, Series)

    


    def test_astype_float(self):
        t = torch.tensor([True, False, True])
        s = Series(t)
        result = s.astype(float)
        assert isinstance(result, Series)
        assert result.values.dtype == torch.float64
        assert result.values.tolist() == [1.0, 0.0, 1.0]

    def test_index_property(self):
        t = torch.tensor([1., 2.])
        idx = Index({'InstrumentID': torch.tensor([0, 1])})
        s = Series(t, index=idx)
        assert s.index is idx

    def test_index_none_by_default(self):
        s = Series(torch.tensor([1., 2.]))
        assert s.index is None


class TestIndex:
    def test_get_level_values(self):
        idx = Index({'InstrumentID': torch.tensor([0, 1, 2])})
        result = idx.get_level_values('InstrumentID')
        assert torch.equal(result, torch.tensor([0, 1, 2]))

    def test_get_level_values_missing_key(self):
        idx = Index({'InstrumentID': torch.tensor([0, 1])})
        with pytest.raises(KeyError):
            idx.get_level_values('nonexistent')

class TestDataFrame:
    def test_creation_from_dict(self):
        t_a = torch.tensor([1., 2.], dtype=torch.double)
        t_b = torch.tensor([3., 4.], dtype=torch.double)
        df = DataFrame({'a': t_a, 'b': t_b})
        assert torch.equal(df._data['a'], t_a)
        assert torch.equal(df._data['b'], t_b)

    def test_getattr_column_access(self):
        t = torch.tensor([100., 200.], dtype=torch.double)
        df = DataFrame({'price': t})
        result = df.price
        assert isinstance(result, Series)
        assert torch.equal(result.values, t)

    def test_getitem_column_access(self):
        t = torch.tensor([100., 200.], dtype=torch.double)
        df = DataFrame({'price': t})
        result = df['price']
        assert isinstance(result, Series)
        assert torch.equal(result.values, t)

    def test_getattr_nonexistent_raises(self):
        df = DataFrame({'price': torch.tensor([100.])})
        with pytest.raises(AttributeError):
            _ = df.nonexistent

    def test_getattr_underscore_passthrough(self):
        df = DataFrame({'price': torch.tensor([100.])})
        with pytest.raises(AttributeError):
            _ = df._private_attr

    def test_groupby_callable(self):
        df = DataFrame({'InstrumentID': torch.tensor([0, 1]), 'price': torch.tensor([100., 200.], dtype=torch.double)})
        assert callable(df.groupby)

    def test_columns_property(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        assert df.columns == ['a', 'b']

    def test_shape_property(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.]), 'b': torch.tensor([4., 5., 6.])})
        assert df.shape == (3, 2)

    def test_len(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        assert len(df) == 3

    def test_dtypes_property(self):
        df = DataFrame({'a': torch.tensor([1., 2.], dtype=torch.float64)})
        assert 'a' in df.dtypes
        assert df.dtypes['a'] == torch.float64

    def test_index_get(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        idx = df.index
        assert len(idx) == 3

    def test_index_set(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        new_idx = torch.tensor([10, 20, 30], dtype=torch.long)
        df.index = new_idx
        assert (df.index == new_idx).all()

    def test_setitem(self):
        df = DataFrame({'a': torch.tensor([1., 2.])})
        df['b'] = torch.tensor([3., 4.])
        assert df.columns == ['a', 'b']
        assert df['b'].values.tolist() == [3., 4.]

    def test_setitem_overwrite(self):
        df = DataFrame({'a': torch.tensor([1., 2.])})
        df['a'] = torch.tensor([9., 8.])
        assert df['a'].values.tolist() == [9., 8.]

    def test_head_default(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3., 4., 5., 6.])})
        h = df.head()
        assert len(h) == 5

    def test_head_n(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3., 4., 5., 6.])})
        h = df.head(2)
        assert len(h) == 2
        assert h['a'].values.tolist() == [1., 2.]

    def test_tail(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3., 4., 5.])})
        t = df.tail(2)
        assert len(t) == 2
        assert t['a'].values.tolist() == [4., 5.]

    def test_drop_columns(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        d = df.drop(columns=['b'])
        assert d.columns == ['a']
        assert 'b' not in d._data

    def test_rename_columns(self):
        df = DataFrame({'a': torch.tensor([1., 2.])})
        r = df.rename(columns={'a': 'x'})
        assert r.columns == ['x']
        assert 'a' not in r._data

    def test_empty_dataframe(self):
        df = DataFrame({})
        assert df.columns == []
        assert df.shape == (0, 0)
        assert len(df) == 0
class TestGroupBy:
    def test_getitem_returns_groupby_column(self):
        df = DataFrame({'InstrumentID': torch.tensor([0,1], dtype=torch.long),
                        'price': torch.tensor([100., 200.], dtype=torch.double)})
        gb = GroupBy(df, 'InstrumentID')
        result = gb['price']
        assert isinstance(result, GroupByColumn)

class TestGroupByColumn:
    def test_resample_returns_resampler(self):
        df = DataFrame({'InstrumentID': torch.tensor([0,1], dtype=torch.long),
                        'price': torch.tensor([100., 200.], dtype=torch.double)})
        gbc = GroupBy(df, 'InstrumentID')['price']
        result = gbc.resample('1D')
        assert isinstance(result, Resampler)

class TestResampler:
    def _make_resampler(self):
        df = DataFrame({
            'InstrumentID': torch.tensor([0,0,0,1,1,1], dtype=torch.long),
            'price': torch.tensor([100.,105.,102.,200.,198.,210.], dtype=torch.double)
        })
        return df.groupby('InstrumentID')['price'].resample('1D')

    def test_first_returns_series(self):
        from xpandas.wrappers import Series
        r = self._make_resampler()
        result = r.first()
        assert isinstance(result, Series)
        assert result.values.tolist() == [100., 200.]

    def test_max_returns_series(self):
        from xpandas.wrappers import Series
        r = self._make_resampler()
        result = r.max()
        assert isinstance(result, Series)
        assert result.values.tolist() == [105., 210.]

    def test_min_returns_series(self):
        from xpandas.wrappers import Series
        r = self._make_resampler()
        result = r.min()
        assert isinstance(result, Series)
        assert result.values.tolist() == [100., 198.]

    def test_last_returns_series(self):
        from xpandas.wrappers import Series
        r = self._make_resampler()
        result = r.last()
        assert isinstance(result, Series)
        assert result.values.tolist() == [102., 210.]

    def test_ohlc_cache_single_call(self):
        r = self._make_resampler()
        assert r._cache is None
        first_result = r.first()
        assert r._cache is not None, 'Cache should be populated after first()'
        max_result = r.max()
        assert first_result.values.tolist() == [100., 200.]
        assert max_result.values.tolist() == [105., 210.]

    def test_series_has_index(self):
        r = self._make_resampler()
        o = r.first()
        assert o.index is not None
        keys = o.index.get_level_values('InstrumentID')
        assert keys.tolist() == [0, 1]

    def test_full_ohlc_chain(self):
        df = DataFrame({
            'InstrumentID': torch.tensor([0,0,0,1,1,1], dtype=torch.long),
            'price': torch.tensor([100.,105.,102.,200.,198.,210.], dtype=torch.double)
        })
        gs = df.groupby('InstrumentID')
        o = gs['price'].resample('1D').first()
        h = gs['price'].resample('1D').max()
        l = gs['price'].resample('1D').min()
        c = gs['price'].resample('1D').last()
        assert o.values.tolist() == [100., 200.]
        assert h.values.tolist() == [105., 210.]
        assert l.values.tolist() == [100., 198.]
        assert c.values.tolist() == [102., 210.]
        assert o.index.get_level_values('InstrumentID').tolist() == [0, 1]


class TestSeriesWindow:
    def test_rolling_returns_rolling_obj(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.rolling(3)
        assert type(r).__name__ == 'Rolling'

    def test_rolling_mean(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.rolling(3).mean()
        assert isinstance(r, Series)
        assert len(r.values) == 5

    def test_rolling_sum(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.rolling(3).sum()
        assert isinstance(r, Series)

    def test_rolling_std(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.rolling(3).std()
        assert isinstance(r, Series)

    def test_rolling_min(self):
        s = Series(torch.tensor([3., 1., 4., 1., 5.], dtype=torch.double))
        r = s.rolling(3).min()
        assert isinstance(r, Series)

    def test_rolling_max(self):
        s = Series(torch.tensor([3., 1., 4., 1., 5.], dtype=torch.double))
        r = s.rolling(3).max()
        assert isinstance(r, Series)

    def test_ewm_returns_ewm_obj(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        e = s.ewm(span=3)
        assert type(e).__name__ == 'EWM'

    def test_ewm_mean(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.ewm(span=3).mean()
        assert isinstance(r, Series)
        assert len(r.values) == 5

    def test_expanding_returns_expanding_obj(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        e = s.expanding()
        assert type(e).__name__ == 'Expanding'

    def test_expanding_sum(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.expanding().sum()
        assert isinstance(r, Series)
        # expanding sum: [1, 3, 6]
        assert r.values.tolist() == [1., 3., 6.]

    def test_expanding_mean(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.expanding().mean()
        assert isinstance(r, Series)
        # expanding mean: [1, 1.5, 2]
        assert abs(r.values[2].item() - 2.0) < 1e-9


class TestSeriesStats:
    def test_series_abs(self):
        s = Series(torch.tensor([-1., 2., -3.], dtype=torch.double))
        r = s.abs()
        assert isinstance(r, Series)
        assert r.values.tolist() == [1., 2., 3.]

    def test_series_log(self):
        s = Series(torch.tensor([1., 2.718281828], dtype=torch.double))
        r = s.log()
        assert isinstance(r, Series)
        # log(1) ≈ 0, log(e) ≈ 1
        assert abs(r.values[0].item()) < 1e-6
        assert abs(r.values[1].item() - 1.0) < 1e-4

    def test_series_zscore(self):
        s = Series(torch.tensor([1., 2., 3., 4., 5.], dtype=torch.double))
        r = s.zscore()
        assert isinstance(r, Series)
        assert abs(r.values.mean().item()) < 1e-6  # mean ~0

    def test_series_rank(self):
        s = Series(torch.tensor([3., 1., 2.], dtype=torch.double))
        r = s.rank()
        assert isinstance(r, Series)
        # rank returns average rank; [3,1,2] → ranks [3,1,2]
        assert r.values.tolist() == [3., 1., 2.]

    def test_series_fillna(self):
        s = Series(torch.tensor([1., float('nan'), 3.], dtype=torch.double))
        r = s.fillna(0.0)
        assert isinstance(r, Series)
        assert r.values[1].item() == 0.0

    def test_series_shift(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.shift(1)
        assert isinstance(r, Series)
        assert len(r.values) == 3

    def test_series_pct_change(self):
        s = Series(torch.tensor([1., 2., 4.], dtype=torch.double))
        r = s.pct_change(1)
        assert isinstance(r, Series)
        assert len(r.values) == 3

    def test_series_cumsum(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.cumsum()
        assert isinstance(r, Series)
        assert r.values.tolist() == [1., 3., 6.]

    def test_series_cumprod(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.cumprod()
        assert isinstance(r, Series)
        assert r.values.tolist() == [1., 2., 6.]

    def test_series_clip(self):
        s = Series(torch.tensor([0., 1., 2., 3.], dtype=torch.double))
        r = s.clip(0.5, 2.5)
        assert isinstance(r, Series)
        assert r.values.tolist() == [0.5, 1., 2., 2.5]

    def test_series_where(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        cond = Series(torch.tensor([True, False, True]))
        other = Series(torch.tensor([0., 0., 0.], dtype=torch.double))
        r = s.where(cond, other)
        assert isinstance(r, Series)
        # where(cond, other): keep self where cond is True, use other where cond is False
        assert r.values.tolist() == [1., 0., 3.]

    def test_series_mask(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        mask_s = Series(torch.tensor([False, True, False]))
        r = s.mask(mask_s, 0.0)
        assert isinstance(r, Series)
        assert r.values[1].item() == 0.0

    def test_series_mean(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert abs(s.mean() - 2.0) < 1e-9

    def test_series_std(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert s.std() > 0

    def test_series_sum(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert s.sum() == 6.0

    def test_series_min_max(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        assert s.min() == 1.0
        assert s.max() == 3.0


class TestDataFrameSorting:
    def test_sort_values_ascending(self):
        df = DataFrame({'a': torch.tensor([3., 1., 2.], dtype=torch.float64), 'b': torch.tensor([30., 10., 20.], dtype=torch.float64)})
        s = df.sort_values('a')
        assert isinstance(s, DataFrame)
        assert s['a'].values.tolist() == [1., 2., 3.]

    def test_sort_values_descending(self):
        df = DataFrame({'a': torch.tensor([3., 1., 2.], dtype=torch.float64)})
        s = df.sort_values('a', ascending=False)
        assert s['a'].values.tolist() == [3., 2., 1.]

    def test_sort_values_returns_new_df(self):
        df = DataFrame({'a': torch.tensor([3., 1., 2.], dtype=torch.float64)})
        s = df.sort_values('a')
        # Original unchanged
        assert df['a'].values.tolist() == [3., 1., 2.]

    def test_reset_index(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        df.index = torch.tensor([10, 20, 30], dtype=torch.long)
        r = df.reset_index()
        assert isinstance(r, DataFrame)
        # index should be reset to 0..n-1
        assert (r.index == torch.arange(3)).all()

    def test_set_index(self):
        df = DataFrame({'key': torch.tensor([10, 20, 30], dtype=torch.long), 'val': torch.tensor([1., 2., 3.])})
        r = df.set_index('key')
        assert isinstance(r, DataFrame)
        # 'key' column should be removed from _data and become index
        assert 'key' not in r._data
        assert len(r['val'].values) == 3

    def test_value_counts(self):
        s = Series(torch.tensor([1, 2, 1, 3, 2, 1], dtype=torch.long))
        vc = s.value_counts()
        assert isinstance(vc, Series)
        # Should have counts for each unique value
        assert len(vc.values) == 3
class TestDataFrameIndexing:
    def test_getitem_str_backward_compat(self):
        """Existing behavior must still work: df['col'] returns Series"""
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        r = df['a']
        assert isinstance(r, Series)
    
    def test_getitem_bool_tensor_mask(self):
        """New: df[bool_tensor] filters rows"""
        df = DataFrame({'a': torch.tensor([1., 2., 3.]), 'b': torch.tensor([4., 5., 6.])})
        mask = torch.tensor([True, False, True])
        r = df[mask]
        assert isinstance(r, DataFrame)
        assert len(r) == 2
class TestApplyMap:
    def test_series_apply(self):
        s = Series(torch.tensor([1., 4., 9.], dtype=torch.double))
        r = s.apply(torch.sqrt)
        assert isinstance(r, Series)
        assert abs(r.values[0].item() - 1.0) < 1e-6
        assert abs(r.values[1].item() - 2.0) < 1e-6

    def test_series_map_function(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.map(lambda x: x * 2)
        assert isinstance(r, Series)
        assert r.values.tolist() == [2., 4., 6.]

    def test_series_agg_mean(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.agg('mean')
        assert isinstance(r, float) or isinstance(r, torch.Tensor)
        assert abs(float(r) - 2.0) < 1e-9

    def test_series_transform(self):
        s = Series(torch.tensor([1., 2., 3.], dtype=torch.double))
        r = s.transform(lambda x: x * 2)
        assert isinstance(r, Series)
        assert len(r.values) == len(s.values)  # same shape

    def test_df_apply_column_wise(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.], dtype=torch.double),
                        'b': torch.tensor([4., 5., 6.], dtype=torch.double)})
        r = df.apply(lambda col: col * 2)
        assert isinstance(r, DataFrame)
        assert r['a'].values.tolist() == [2., 4., 6.]

    def test_df_applymap(self):
        df = DataFrame({'a': torch.tensor([1., 2.], dtype=torch.double)})
        r = df.applymap(lambda x: x + 1)
        assert isinstance(r, DataFrame)
        # Each element incremented by 1
        assert r['a'].values[0].item() == 2.0

    def test_df_agg_mean(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.], dtype=torch.double)})
        r = df.agg('mean')
        # Returns a dict or Series with mean per column
        assert isinstance(r, (dict, Series, DataFrame))

    def test_df_pipe(self):
        df = DataFrame({'a': torch.tensor([1., 2.], dtype=torch.double)})
        r = df.pipe(lambda d: d)
        assert isinstance(r, DataFrame)
        assert r is df or r._data == df._data

    def test_series_pipe(self):
        s = Series(torch.tensor([1., 2.], dtype=torch.double))
        r = s.pipe(lambda x: x)
        assert isinstance(r, Series)
        assert r.values.tolist() == [1., 2.]
    
    def test_getitem_series_mask(self):
        """New: df[Series_of_bool] filters rows"""
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        mask = Series(torch.tensor([True, False, True]))
        r = df[mask]
        assert isinstance(r, DataFrame)
        assert len(r) == 2
    
    def test_getitem_list_of_cols(self):
        """New: df[['a', 'b']] returns sub-DataFrame"""
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.]), 'c': torch.tensor([5., 6.])})
        r = df[['a', 'b']]
        assert isinstance(r, DataFrame)
        assert r.columns == ['a', 'b']
    
    def test_iloc_single_row(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.]), 'b': torch.tensor([4., 5., 6.])})
        r = df.iloc[0]
        # Single row as dict-like or Series
        assert isinstance(r, dict) or isinstance(r, Series)
    
    def test_iloc_slice(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3., 4.])})
        r = df.iloc[1:3]
        assert isinstance(r, DataFrame)
        assert len(r) == 2
        assert r['a'].values.tolist() == [2., 3.]
    
    def test_iloc_column_slice(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.]), 'b': torch.tensor([4., 5., 6.])})
        r = df.iloc[:, 0]
        assert isinstance(r, Series)
    
    def test_loc_by_index(self):
        df = DataFrame({'a': torch.tensor([10., 20., 30.])})
        r = df.loc[0]
        # loc with integer returns row
        assert r is not None


class TestMergeConcatConvert:
    def test_merge_inner(self):
        left = DataFrame({'id': torch.tensor([1., 2., 3.]), 'val': torch.tensor([10., 20., 30.])})
        right = DataFrame({'id': torch.tensor([2., 3., 4.]), 'score': torch.tensor([0.2, 0.3, 0.4])})
        r = left.merge(right, on='id')
        assert isinstance(r, DataFrame)
        assert len(r) == 2  # ids 2, 3 match
        assert 'val' in r.columns
        assert 'score' in r.columns

    def test_join_index_align(self):
        left = DataFrame({'a': torch.tensor([1., 2., 3.])})
        right = DataFrame({'b': torch.tensor([4., 5., 6.])})
        r = left.join(right)
        assert isinstance(r, DataFrame)
        assert 'a' in r.columns and 'b' in r.columns
        assert len(r) == 3

    def test_concat_vertical(self):
        import xpandas as pd
        df1 = DataFrame({'a': torch.tensor([1., 2.])})
        df2 = DataFrame({'a': torch.tensor([3., 4.])})
        r = pd.concat([df1, df2], axis=0)
        assert isinstance(r, DataFrame)
        assert len(r) == 4

    def test_concat_horizontal(self):
        import xpandas as pd
        df1 = DataFrame({'a': torch.tensor([1., 2.])})
        df2 = DataFrame({'b': torch.tensor([3., 4.])})
        r = pd.concat([df1, df2], axis=1)
        assert isinstance(r, DataFrame)
        assert 'a' in r.columns and 'b' in r.columns

    def test_to_dict(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        d = df.to_dict()
        assert d == {'a': [1.0, 2.0], 'b': [3.0, 4.0]}

    def test_to_dict_records(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        r = df.to_dict(orient='records')
        assert len(r) == 2
        assert r[0]['a'] == 1.0 and r[0]['b'] == 3.0

    def test_to_numpy(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        arr = df.to_numpy()
        import numpy as np
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_describe(self):
        df = DataFrame({'a': torch.tensor([1., 2., 3.])})
        d = df.describe()
        assert 'a' in d
        assert abs(d['a']['mean'] - 2.0) < 1e-6
        assert d['a']['count'] == 3.0

    def test_info_runs(self):
        df = DataFrame({'a': torch.tensor([1., 2.]), 'b': torch.tensor([3., 4.])})
        result = df.info()
        assert result is None  # info() returns None, just prints