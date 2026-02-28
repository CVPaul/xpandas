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

    def test_sub_operator_returns_tensor(self):
        a = Series(torch.tensor([1., 0.], dtype=torch.double))
        b = Series(torch.tensor([0., 1.], dtype=torch.double))
        result = a - b
        assert isinstance(result, torch.Tensor), f'Expected Tensor, got {type(result)}'
        assert not isinstance(result, Series), 'Must be raw Tensor, not Series'
        assert result.tolist() == [1.0, -1.0]

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