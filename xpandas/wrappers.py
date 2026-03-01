class _iLocIndexer:
    def __init__(self, df):
        self._df = df
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # df.iloc[row_idx, col_idx]
            row_idx, col_idx = idx
            if isinstance(col_idx, int):
                col_name = list(self._df._data.keys())[col_idx]
                col_data = self._df._data[col_name]
                return Series(col_data[row_idx] if not isinstance(row_idx, slice) else col_data[row_idx])
            # Fall back to row selection
            row_idx = idx[0]
        
        if isinstance(idx, int):
            # Single row → dict
            return {k: v[idx].item() for k, v in self._df._data.items()}
        elif isinstance(idx, slice):
            # Row slice → DataFrame
            return DataFrame({k: v[idx] for k, v in self._df._data.items()})
        else:
            raise IndexError(f"Unsupported iloc index: {type(idx)}")
class _LocIndexer:
    def __init__(self, df):
        self._df = df
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # loc with integer: return row as dict
            return {k: v[idx].item() for k, v in self._df._data.items()}
        elif isinstance(idx, slice):
            return DataFrame({k: v[idx] for k, v in self._df._data.items()})
        else:
            raise IndexError(f"Unsupported loc index: {type(idx)}")

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
        other_data = other._data if isinstance(other, Series) else torch.tensor(other, dtype=self._data.dtype, device=self._data.device) if not torch.is_tensor(other) else other
        return Series(torch.ops.xpandas.compare_gt(self._data, other_data))

    def __lt__(self, other):
        other_data = other._data if isinstance(other, Series) else torch.tensor(other, dtype=self._data.dtype, device=self._data.device) if not torch.is_tensor(other) else other
        return Series(torch.ops.xpandas.compare_lt(self._data, other_data))

    def _extract(self, other):
        """Extract underlying tensor from other if it's a Series."""
        return other._data if isinstance(other, Series) else other

    def __sub__(self, other):
        return Series(self._data - self._extract(other))

    def __rsub__(self, other):
        return Series(self._extract(other) - self._data)

    def __add__(self, other):
        return Series(self._data + self._extract(other))

    def __radd__(self, other):
        return Series(self._extract(other) + self._data)

    def __mul__(self, other):
        return Series(self._data * self._extract(other))

    def __rmul__(self, other):
        return Series(self._extract(other) * self._data)

    def __truediv__(self, other):
        return Series(self._data / self._extract(other))

    def __rtruediv__(self, other):
        return Series(self._extract(other) / self._data)

    def __neg__(self):
        return Series(-self._data)

    def __abs__(self):
        return Series(self._data.abs())

    def __pow__(self, other):
        return Series(self._data ** self._extract(other))

    def __mod__(self, other):
        return Series(self._data % self._extract(other))

    def __and__(self, other):
        return Series(self._data & self._extract(other))

    def __or__(self, other):
        return Series(self._data | self._extract(other))

    def __eq__(self, other):
        return Series(self._data == self._extract(other))

    def __ne__(self, other):
        return Series(self._data != self._extract(other))

    def __ge__(self, other):
        return Series(self._data >= self._extract(other))

    def __le__(self, other):
        return Series(self._data <= self._extract(other))
    def astype(self, dtype):
        if dtype is float or dtype == float:
            return Series(torch.ops.xpandas.bool_to_float(self._data))
        raise NotImplementedError(f"astype({dtype}) not supported")

    # Statistical/math methods wrapping C++ ops
    def abs(self):
        return Series(torch.ops.xpandas.abs_(self._data))

    def log(self):
        return Series(torch.ops.xpandas.log_(self._data))

    def zscore(self):
        return Series(torch.ops.xpandas.zscore(self._data))

    def rank(self):
        return Series(torch.ops.xpandas.rank(self._data))

    def fillna(self, value):
        return Series(torch.ops.xpandas.fillna(self._data, float(value)))

    def shift(self, periods=1):
        return Series(torch.ops.xpandas.shift(self._data, int(periods)))

    def pct_change(self, periods=1):
        return Series(torch.ops.xpandas.pct_change(self._data, int(periods)))

    def apply(self, func):
        """Apply a function to the underlying tensor (or element-wise)."""
        result = func(self._data)
        if isinstance(result, Series):
            return result
        return Series(result)

    def map(self, func):
        """Apply a function element-wise or to the whole tensor."""
        if callable(func):
            try:
                result = func(self._data)
                if torch.is_tensor(result):
                    return Series(result)
                result = torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype)
                return Series(result)
            except Exception:
                result = torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype)
                return Series(result)
        raise TypeError(f"map requires callable, got {type(func)}")

    def agg(self, func):
        """Aggregate Series with a function or named operation."""
        if func == 'mean':
            return self._data.mean().item()
        elif func == 'sum':
            return self._data.sum().item()
        elif func == 'min':
            return self._data.min().item()
        elif func == 'max':
            return self._data.max().item()
        elif func == 'std':
            return self._data.std().item()
        elif callable(func):
            return func(self._data)
        raise ValueError(f"Unknown agg function: {func}")

    def transform(self, func):
        """Apply function and return same-shape Series."""
        result = func(self._data)
        if torch.is_tensor(result):
            return Series(result)
        return Series(torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype))

    def pipe(self, func, *args, **kwargs):
        """Pipe self through a function."""
        return func(self, *args, **kwargs)
    def cumsum(self):
        return Series(torch.ops.xpandas.cumsum(self._data))

    def cumprod(self):
        return Series(torch.ops.xpandas.cumprod(self._data))

    def clip(self, lower, upper):
        return Series(torch.ops.xpandas.clip(self._data, float(lower), float(upper)))

    def where(self, cond, other):
        cond_data = cond._data if isinstance(cond, Series) else cond
        other_data = other._data if isinstance(other, Series) else other
        return Series(torch.ops.xpandas.where_(cond_data, self._data, other_data))

    def mask(self, cond, value):
        cond_data = cond._data if isinstance(cond, Series) else cond
        return Series(torch.ops.xpandas.masked_fill(self._data, cond_data, float(value)))

    # Pure torch aggregation methods
    def mean(self):
        return self._data.mean().item()

    def std(self):
        return self._data.std().item()

    def sum(self):
        return self._data.sum().item()

    def min(self):
        return self._data.min().item()

    def max(self):
        return self._data.max().item()

    def rolling(self, window: int):
        return Rolling(self, window)

    def ewm(self, span: int):
        return EWM(self, span)

    def expanding(self):
        return Expanding(self)

    def value_counts(self):
        unique_vals, counts = torch.unique(self._data, return_counts=True)
        return Series(counts.to(dtype=torch.long), index=Index({'values': unique_vals}))

class DataFrame:
    def __init__(self, data: dict):
        self._data = dict(data)  # shallow copy
        self._index = None
    
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._data:
            return Series(self._data[name])
        raise AttributeError(f"'DataFrame' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return Series(self._data[key])
        elif isinstance(key, list):
            # List of column names → sub-DataFrame
            return DataFrame({k: self._data[k] for k in key})
        elif isinstance(key, Series):
            # Boolean Series mask → filter rows
            mask = key._data
            return DataFrame({k: v[mask] for k, v in self._data.items()})
        elif torch.is_tensor(key) and key.dtype == torch.bool:
            # Boolean tensor mask → filter rows
            return DataFrame({k: v[key] for k, v in self._data.items()})
        else:
            raise KeyError(f"Unsupported key type: {type(key)}")
    def groupby(self, by: str):
        return GroupBy(self, by)

    @property
    def columns(self):
        return list(self._data.keys())

    @property
    def shape(self):
        if not self._data:
            return (0, 0)
        first = next(iter(self._data.values()))
        return (len(first), len(self._data))

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    @property
    def dtypes(self):
        return {k: v.dtype for k, v in self._data.items()}

    @property
    def index(self):
        if self._index is None:
            if not self._data:
                return torch.arange(0)
            n = len(next(iter(self._data.values())))
            return torch.arange(n)
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    def __setitem__(self, key, value):
        self._data[key] = value

    def head(self, n=5):
        return DataFrame({k: v[:n] for k, v in self._data.items()})

    def tail(self, n=5):
        return DataFrame({k: v[-n:] for k, v in self._data.items()})

    def drop(self, columns=None):
        if columns is None:
            return DataFrame(dict(self._data))
        cols_to_drop = [columns] if isinstance(columns, str) else list(columns)
        return DataFrame({k: v for k, v in self._data.items() if k not in cols_to_drop})

    def rename(self, columns=None):
        if columns is None:
            return DataFrame(dict(self._data))
        return DataFrame({columns.get(k, k): v for k, v in self._data.items()})

    @property
    def iloc(self):
        return _iLocIndexer(self)

    @property
    def loc(self):
        return _LocIndexer(self)

    def sort_values(self, by: str, ascending: bool = True):
        sorted_data = torch.ops.xpandas.sort_by(self._data, by, ascending)
        return DataFrame(sorted_data)

    def reset_index(self, drop: bool = True):
        new_df = DataFrame(dict(self._data))
        new_df._index = None  # Will default to arange
        return new_df

    def set_index(self, keys: str):
        key_tensor = self._data[keys]
        new_data = {k: v for k, v in self._data.items() if k != keys}
        new_df = DataFrame(new_data)
        new_df._index = key_tensor
        return new_df

    def apply(self, func, axis=0):
        """Apply function to each column (axis=0) or row (axis=1)."""
        if axis == 0:
            result = {}
            for k, v in self._data.items():
                r = func(Series(v))
                result[k] = r._data if isinstance(r, Series) else r
            return DataFrame(result)
        else:
            cols = list(self._data.keys())
            n = len(self)
            rows = [{k: self._data[k][i] for k in cols} for i in range(n)]
            results = [func(row) for row in rows]
            if results and isinstance(results[0], dict):
                return DataFrame({k: torch.stack([r[k] for r in results]) for k in results[0]})
            return Series(torch.tensor([float(r) if not torch.is_tensor(r) else r.item() for r in results]))

    def applymap(self, func):
        """Apply function element-wise to all cells."""
        new_data = {}
        for k, v in self._data.items():
            new_data[k] = torch.tensor([func(x.item()) for x in v], dtype=v.dtype)
        return DataFrame(new_data)

    def agg(self, func):
        """Aggregate each column."""
        if isinstance(func, str):
            if func == 'mean':
                return {k: v.mean().item() for k, v in self._data.items()}
            elif func == 'sum':
                return {k: v.sum().item() for k, v in self._data.items()}
            elif func == 'min':
                return {k: v.min().item() for k, v in self._data.items()}
            elif func == 'max':
                return {k: v.max().item() for k, v in self._data.items()}
        elif callable(func):
            return {k: func(v) for k, v in self._data.items()}
        raise ValueError(f"Unknown agg: {func}")

    def pipe(self, func, *args, **kwargs):
        """Pass DataFrame as first arg to func."""
        return func(self, *args, **kwargs)

    def merge(self, other, on: str, how: str = 'inner'):
        """Simple key-equality merge on a single column."""
        left_key = self._data[on]
        right_key = other._data[on]
        # Build index lookup for right side
        right_index = {}
        for i, k in enumerate(right_key.tolist()):
            right_index[k] = i
        # Find matching rows
        left_rows = []
        right_rows = []
        for i, k in enumerate(left_key.tolist()):
            if k in right_index:
                left_rows.append(i)
                right_rows.append(right_index[k])
        if not left_rows:
            # Return empty DataFrame with all columns
            cols = list(self._data.keys()) + [k for k in other._data.keys() if k != on]
            return DataFrame({c: torch.tensor([], dtype=self._data.get(c, list(other._data.values())[0]).dtype) for c in cols})
        left_idx = torch.tensor(left_rows, dtype=torch.long)
        right_idx = torch.tensor(right_rows, dtype=torch.long)
        result = {}
        for k, v in self._data.items():
            result[k] = v[left_idx]
        for k, v in other._data.items():
            if k != on:
                result[k] = v[right_idx]
        return DataFrame(result)

    def join(self, other, on: str = None, how: str = 'left'):
        """Alias for merge with left join semantics."""
        if on is not None:
            return self.merge(other, on=on, how=how)
        # Join on index (row-by-row alignment)
        n = min(len(self), len(other))
        result = {k: v[:n] for k, v in self._data.items()}
        for k, v in other._data.items():
            if k not in result:
                result[k] = v[:n]
        return DataFrame(result)

    def to_numpy(self):
        """Convert to numpy array (columns stacked)."""
        import numpy as np
        arrays = [v.numpy() for v in self._data.values()]
        return np.column_stack(arrays) if arrays else np.array([])

    def to_dict(self, orient: str = 'dict'):
        """Convert to Python dict."""
        if orient == 'dict':
            return {k: v.tolist() for k, v in self._data.items()}
        elif orient == 'list':
            return {k: v.tolist() for k, v in self._data.items()}
        elif orient == 'records':
            cols = list(self._data.keys())
            n = len(self)
            return [{c: self._data[c][i].item() for c in cols} for i in range(n)]
        raise ValueError(f"Unknown orient: {orient}")

    def describe(self):
        """Return summary statistics as a dict of dicts."""
        result = {}
        for k, v in self._data.items():
            v_float = v.float()
            result[k] = {
                'count': float(len(v)),
                'mean': v_float.mean().item(),
                'std': v_float.std().item(),
                'min': v_float.min().item(),
                'max': v_float.max().item(),
            }
        return result

    def info(self):
        """Print a summary of the DataFrame. Returns None."""
        print(f"DataFrame: {self.shape[0]} rows x {self.shape[1]} cols")
        for k, v in self._data.items():
            print(f"  {k}: {v.dtype}, {len(v)} non-null")
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


class Rolling:
    """Proxy for series.rolling(window).method()"""
    def __init__(self, series, window: int):
        self._series = series
        self._window = window

    def mean(self):
        return Series(torch.ops.xpandas.rolling_mean(self._series._data, self._window))

    def sum(self):
        return Series(torch.ops.xpandas.rolling_sum(self._series._data, self._window))

    def std(self):
        return Series(torch.ops.xpandas.rolling_std(self._series._data, self._window))

    def min(self):
        return Series(torch.ops.xpandas.rolling_min(self._series._data, self._window))

    def max(self):
        return Series(torch.ops.xpandas.rolling_max(self._series._data, self._window))


class EWM:
    """Proxy for series.ewm(span=n).method()"""
    def __init__(self, series, span: int):
        self._series = series
        self._span = span

    def mean(self):
        return Series(torch.ops.xpandas.ewm_mean(self._series._data, self._span))


class Expanding:
    """Proxy for series.expanding().method()"""
    def __init__(self, series):
        self._series = series

    def sum(self):
        return Series(torch.ops.xpandas.cumsum(self._series._data))

    def mean(self):
        # expanding mean: cumsum / arange(1..n)
        data = self._series._data
        cs = torch.ops.xpandas.cumsum(data)
        counts = torch.arange(1, len(data) + 1, dtype=data.dtype, device=data.device)
        return Series(cs / counts)
