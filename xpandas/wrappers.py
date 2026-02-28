
import torch




class Index:
    def __init__(self, levels: dict):
        self._levels = levels

    def get_level_values(self, name: str):
        return self._levels[name]

class Series:
    def __init__(self, data: torch.Tensor, index=None):
        self._data = data
        self._index = index

    @property
    def values(self):
        return self._data

    @property
    def index(self):
        return self._index

    def __gt__(self, other):
        other_data = other._data if isinstance(other, Series) else other
        return Series(torch.ops.xpandas.compare_gt(self._data, other_data))

    def __lt__(self, other):
        other_data = other._data if isinstance(other, Series) else other
        return Series(torch.ops.xpandas.compare_lt(self._data, other_data))

    def __sub__(self, other):
        other_data = other._data if isinstance(other, Series) else other
        return self._data - other_data  # raw Tensor, NOT Series

    def astype(self, dtype):
        if dtype is float or dtype == float:
            return Series(torch.ops.xpandas.bool_to_float(self._data))
        raise NotImplementedError(f"astype({dtype}) not supported")

class DataFrame:
    def __init__(self, data: dict):
        self._data = dict(data)  # shallow copy
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._data:
            return Series(self._data[name])
        raise AttributeError(f"'DataFrame' object has no attribute '{name}'")
    
    def __getitem__(self, name: str):
        return Series(self._data[name])
    
    def groupby(self, by: str):
        return GroupBy(self, by)

class GroupBy:
    def __init__(self, df: DataFrame, by: str):
        self._df = df
        self._by = by

    def __getitem__(self, col: str):
        return GroupByColumn(self._df, self._by, col)

class GroupByColumn:
    def __init__(self, df, by: str, col: str):
        self._df = df
        self._by = by
        self._col = col

    def resample(self, freq: str):
        return Resampler(self._df, self._by, self._col, freq)

class Resampler:
    def __init__(self, df, by: str, col: str, freq: str):
        self._key = df._data[by]
        self._value = df._data[col]
        self._cache = None

    def _compute_ohlc(self):
        if self._cache is None:
            self._cache = torch.ops.xpandas.groupby_resample_ohlc(self._key, self._value)
        return self._cache

    def first(self) -> Series:
        cache = self._compute_ohlc()
        return Series(cache[1], index=Index({'InstrumentID': cache[0]}))

    def max(self) -> Series:
        cache = self._compute_ohlc()
        return Series(cache[2], index=Index({'InstrumentID': cache[0]}))

    def min(self) -> Series:
        cache = self._compute_ohlc()
        return Series(cache[3], index=Index({'InstrumentID': cache[0]}))

    def last(self) -> Series:
        cache = self._compute_ohlc()
        return Series(cache[4], index=Index({'InstrumentID': cache[0]}))
