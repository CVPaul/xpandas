"""
xpandas.wrappers -- High-level pandas-compatible wrappers around ``torch.ops.xpandas.*`` C++ ops.

These classes provide a familiar ``DataFrame`` / ``Series`` / ``GroupBy`` /
``Rolling`` / ``EWM`` / ``Expanding`` API that mirrors pandas.  Every method
dispatches to a registered ``TORCH_LIBRARY`` C++ op, so the full pipeline is
``torch.jit.script``-compatible and can run in pure C++ inference.

Usage::

    import xpandas as pd

    df = pd.DataFrame({"price": price_tensor, "volume": vol_tensor})
    signal = df.price.rolling(20).mean() - df.price.rolling(60).mean()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_tensor(value: Any, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Convert a scalar or Series to a Tensor, matching *dtype* and *device*."""
    if isinstance(value, Series):
        return value._data
    if torch.is_tensor(value):
        return value
    return torch.tensor(value, dtype=dtype, device=device)


def _validate_1d_double(t: Tensor, op_name: str) -> None:
    """Raise a friendly error if *t* is not a 1-D float64 tensor."""
    if t.dim() != 1:
        raise ValueError(
            f"{op_name}: expected a 1-D Series (got {t.dim()}-D tensor). "
            "xpandas ops operate on flat column vectors."
        )
    if t.dtype != torch.float64:
        raise TypeError(
            f"{op_name}: expected float64 (double) dtype, got {t.dtype}. "
            "Hint: use `series.to(torch.float64)` or construct tensors with "
            "`dtype=torch.double`."
        )


# ---------------------------------------------------------------------------
# _iLocIndexer / _LocIndexer
# ---------------------------------------------------------------------------

class _iLocIndexer:
    """Integer-location based indexer for :class:`DataFrame`.

    Accessed via ``df.iloc[...]``.  Supports integer indices, slices,
    and ``(row, col)`` tuples.
    """

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            if isinstance(col_idx, int):
                col_name = list(self._df._data.keys())[col_idx]
                col_data = self._df._data[col_name]
                return Series(col_data[row_idx] if not isinstance(row_idx, slice) else col_data[row_idx])
            # Fall back to row selection
            row_idx = idx[0]

        if isinstance(idx, int):
            return {k: v[idx].item() for k, v in self._df._data.items()}
        elif isinstance(idx, slice):
            return DataFrame({k: v[idx] for k, v in self._df._data.items()})
        else:
            raise IndexError(f"Unsupported iloc index type: {type(idx).__name__}")


class _LocIndexer:
    """Label-based indexer for :class:`DataFrame`.

    Accessed via ``df.loc[...]``.  Currently supports integer and slice
    indices (label semantics are simplified to positional).
    """

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, int):
            return {k: v[idx].item() for k, v in self._df._data.items()}
        elif isinstance(idx, slice):
            return DataFrame({k: v[idx] for k, v in self._df._data.items()})
        else:
            raise IndexError(f"Unsupported loc index type: {type(idx).__name__}")


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

class Index:
    """Multi-level index for :class:`Series` results (e.g. after groupby).

    Parameters
    ----------
    levels : dict[str, Tensor]
        Mapping from level name to the tensor of level values.

    Example
    -------
    >>> idx = Index({"InstrumentID": torch.tensor([0, 1, 2])})
    >>> idx.get_level_values("InstrumentID")
    tensor([0, 1, 2])
    """

    def __init__(self, levels: Dict[str, Tensor]) -> None:
        self._levels = levels

    def get_level_values(self, name: str) -> Tensor:
        """Return the tensor of values for the given level *name*.

        Raises
        ------
        KeyError
            If *name* is not a valid level.
        """
        if name not in self._levels:
            raise KeyError(
                f"Level '{name}' not found. Available levels: "
                f"{list(self._levels.keys())}"
            )
        return self._levels[name]


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

class Series:
    """A 1-D labeled array backed by a :class:`torch.Tensor`.

    Drop-in replacement for ``pandas.Series`` (subset).  Every method that
    dispatches to a ``torch.ops.xpandas.*`` C++ op is fully
    ``torch.jit.script``-compatible.

    Parameters
    ----------
    data : Tensor
        The underlying 1-D tensor (float64 for numeric, int64 for
        enum-encoded categorical columns).
    index : Index, optional
        An :class:`Index` object for labeled access.

    Example
    -------
    >>> import xpandas as pd, torch
    >>> s = pd.Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
    >>> s.rolling(2).mean()
    """

    def __init__(self, data: Tensor, index: Optional[Index] = None) -> None:
        if not torch.is_tensor(data):
            raise TypeError(
                f"Series requires a torch.Tensor, got {type(data).__name__}. "
                "Hint: wrap your data with torch.tensor(...)."
            )
        self._data = data
        self._index = index

    # -- Properties ---------------------------------------------------------

    @property
    def values(self) -> Tensor:
        """Return the underlying :class:`torch.Tensor`."""
        return self._data

    @property
    def index(self) -> Optional[Index]:
        """Return the :class:`Index`, or ``None`` if unset."""
        return self._index

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the underlying tensor."""
        return self._data.dtype

    @property
    def shape(self) -> torch.Size:
        """Return the shape of the underlying tensor."""
        return self._data.shape

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Series({self._data})"

    # -- Comparison operators (dispatched to C++ ops) -----------------------

    def __gt__(self, other: Any) -> "Series":
        """Element-wise ``>``.  Dispatches to ``xpandas::compare_gt``."""
        other_data = _ensure_tensor(other, dtype=self._data.dtype, device=self._data.device)
        return Series(torch.ops.xpandas.compare_gt(self._data, other_data))

    def __lt__(self, other: Any) -> "Series":
        """Element-wise ``<``.  Dispatches to ``xpandas::compare_lt``."""
        other_data = _ensure_tensor(other, dtype=self._data.dtype, device=self._data.device)
        return Series(torch.ops.xpandas.compare_lt(self._data, other_data))

    # -- Helpers ------------------------------------------------------------

    def _extract(self, other: Any) -> Tensor:
        """Extract the underlying tensor from *other* if it is a Series."""
        return other._data if isinstance(other, Series) else other

    # -- Arithmetic operators -----------------------------------------------

    def __sub__(self, other: Any) -> "Series":
        """Element-wise subtraction."""
        return Series(self._data - self._extract(other))

    def __rsub__(self, other: Any) -> "Series":
        return Series(self._extract(other) - self._data)

    def __add__(self, other: Any) -> "Series":
        """Element-wise addition."""
        return Series(self._data + self._extract(other))

    def __radd__(self, other: Any) -> "Series":
        return Series(self._extract(other) + self._data)

    def __mul__(self, other: Any) -> "Series":
        """Element-wise multiplication."""
        return Series(self._data * self._extract(other))

    def __rmul__(self, other: Any) -> "Series":
        return Series(self._extract(other) * self._data)

    def __truediv__(self, other: Any) -> "Series":
        """Element-wise division."""
        return Series(self._data / self._extract(other))

    def __rtruediv__(self, other: Any) -> "Series":
        return Series(self._extract(other) / self._data)

    def __neg__(self) -> "Series":
        """Unary negation."""
        return Series(-self._data)

    def __abs__(self) -> "Series":
        """Absolute value (Python built-in ``abs()``)."""
        return Series(self._data.abs())

    def __pow__(self, other: Any) -> "Series":
        """Element-wise power."""
        return Series(self._data ** self._extract(other))

    def __mod__(self, other: Any) -> "Series":
        """Element-wise modulo."""
        return Series(self._data % self._extract(other))

    def __and__(self, other: Any) -> "Series":
        """Bitwise AND."""
        return Series(self._data & self._extract(other))

    def __or__(self, other: Any) -> "Series":
        """Bitwise OR."""
        return Series(self._data | self._extract(other))

    def __eq__(self, other: Any) -> "Series":  # type: ignore[override]
        """Element-wise equality."""
        return Series(self._data == self._extract(other))

    def __ne__(self, other: Any) -> "Series":  # type: ignore[override]
        """Element-wise inequality."""
        return Series(self._data != self._extract(other))

    def __ge__(self, other: Any) -> "Series":
        """Element-wise ``>=``."""
        return Series(self._data >= self._extract(other))

    def __le__(self, other: Any) -> "Series":
        """Element-wise ``<=``."""
        return Series(self._data <= self._extract(other))

    # -- Type casting -------------------------------------------------------

    def astype(self, dtype: type) -> "Series":
        """Cast the Series to a new dtype.

        Currently supports ``float`` (dispatches to ``xpandas::bool_to_float``).

        Parameters
        ----------
        dtype : type
            Target Python type.  Only ``float`` is currently supported.

        Returns
        -------
        Series
            A new Series with the converted data.

        Raises
        ------
        NotImplementedError
            If *dtype* is not ``float``.
        """
        if dtype is float or dtype == float:
            return Series(torch.ops.xpandas.bool_to_float(self._data))
        raise NotImplementedError(
            f"astype({dtype}) is not supported. Currently only astype(float) is "
            "available (for boolean → float64 conversion)."
        )

    # -- Statistical / math methods (C++ ops) -------------------------------

    def abs(self) -> "Series":
        """Return element-wise absolute value.

        Dispatches to ``xpandas::abs_``.  Equivalent to ``series.abs()`` in
        pandas.
        """
        _validate_1d_double(self._data, "Series.abs")
        return Series(torch.ops.xpandas.abs_(self._data))

    def log(self) -> "Series":
        """Return element-wise natural logarithm.

        Dispatches to ``xpandas::log_``.  Equivalent to ``np.log(series)``
        in pandas/numpy.

        .. warning::
           Returns NaN for non-positive values.
        """
        _validate_1d_double(self._data, "Series.log")
        return Series(torch.ops.xpandas.log_(self._data))

    def zscore(self) -> "Series":
        """Compute cross-sectional z-score: ``(x - mean) / std``.

        Dispatches to ``xpandas::zscore``.

        Returns
        -------
        Series
            Standardized values with mean ≈ 0, std ≈ 1.
        """
        _validate_1d_double(self._data, "Series.zscore")
        return Series(torch.ops.xpandas.zscore(self._data))

    def rank(self) -> "Series":
        """Rank values using average method (NaN preserved).

        Dispatches to ``xpandas::rank``.  Equivalent to
        ``series.rank(method='average')`` in pandas.

        Returns
        -------
        Series
            1-based ranks with ties averaged.
        """
        _validate_1d_double(self._data, "Series.rank")
        return Series(torch.ops.xpandas.rank(self._data))

    def fillna(self, value: float) -> "Series":
        """Replace NaN values with *value*.

        Dispatches to ``xpandas::fillna``.  Equivalent to
        ``series.fillna(value)`` in pandas.

        Parameters
        ----------
        value : float
            Replacement value.

        Returns
        -------
        Series
            Copy with NaN replaced.
        """
        _validate_1d_double(self._data, "Series.fillna")
        return Series(torch.ops.xpandas.fillna(self._data, float(value)))

    def shift(self, periods: int = 1) -> "Series":
        """Shift data by *periods* positions, filling with NaN.

        Dispatches to ``xpandas::shift``.  Equivalent to
        ``series.shift(periods)`` in pandas.

        Parameters
        ----------
        periods : int
            Positive = shift forward (NaN at start), negative = shift
            backward (NaN at end).

        Returns
        -------
        Series
        """
        _validate_1d_double(self._data, "Series.shift")
        return Series(torch.ops.xpandas.shift(self._data, int(periods)))

    def pct_change(self, periods: int = 1) -> "Series":
        """Compute percentage change between current and *periods*-prior element.

        Dispatches to ``xpandas::pct_change``.  Equivalent to
        ``series.pct_change(periods)`` in pandas.

        Parameters
        ----------
        periods : int
            Offset to compute change over.

        Returns
        -------
        Series
            Fractional change (0.05 = +5%).  First *periods* values are NaN.
        """
        _validate_1d_double(self._data, "Series.pct_change")
        return Series(torch.ops.xpandas.pct_change(self._data, int(periods)))

    def cumsum(self) -> "Series":
        """Cumulative sum.

        Dispatches to ``xpandas::cumsum``.  Equivalent to
        ``series.cumsum()`` in pandas.
        """
        _validate_1d_double(self._data, "Series.cumsum")
        return Series(torch.ops.xpandas.cumsum(self._data))

    def cumprod(self) -> "Series":
        """Cumulative product.

        Dispatches to ``xpandas::cumprod``.  Equivalent to
        ``series.cumprod()`` in pandas.
        """
        _validate_1d_double(self._data, "Series.cumprod")
        return Series(torch.ops.xpandas.cumprod(self._data))

    def clip(self, lower: float, upper: float) -> "Series":
        """Clip values to ``[lower, upper]``.

        Dispatches to ``xpandas::clip``.  Equivalent to
        ``series.clip(lower, upper)`` in pandas.

        Parameters
        ----------
        lower : float
            Minimum threshold.
        upper : float
            Maximum threshold.

        Returns
        -------
        Series
        """
        _validate_1d_double(self._data, "Series.clip")
        return Series(torch.ops.xpandas.clip(self._data, float(lower), float(upper)))

    def where(self, cond: Union["Series", Tensor], other: Union["Series", Tensor, float]) -> "Series":
        """Return values where *cond* is True, else *other*.

        Dispatches to ``xpandas::where_``.  Equivalent to
        ``series.where(cond, other)`` in pandas.

        Parameters
        ----------
        cond : Series or Tensor
            Boolean condition.
        other : Series, Tensor, or float
            Replacement values where *cond* is False.

        Returns
        -------
        Series
        """
        cond_data = cond._data if isinstance(cond, Series) else cond
        other_data = other._data if isinstance(other, Series) else other
        return Series(torch.ops.xpandas.where_(cond_data, self._data, other_data))

    def mask(self, cond: Union["Series", Tensor], value: float) -> "Series":
        """Replace values where *cond* is True with *value*.

        Dispatches to ``xpandas::masked_fill``.  Equivalent to
        ``series.mask(cond, value)`` in pandas.

        Parameters
        ----------
        cond : Series or Tensor
            Boolean mask.
        value : float
            Fill value.

        Returns
        -------
        Series
        """
        cond_data = cond._data if isinstance(cond, Series) else cond
        return Series(torch.ops.xpandas.masked_fill(self._data, cond_data, float(value)))

    # -- Aggregation (pure torch, no C++ custom ops) ------------------------

    def mean(self) -> float:
        """Return the mean of all values."""
        return self._data.mean().item()

    def std(self) -> float:
        """Return the standard deviation (Bessel-corrected)."""
        return self._data.std().item()

    def sum(self) -> float:
        """Return the sum of all values."""
        return self._data.sum().item()

    def min(self) -> float:
        """Return the minimum value."""
        return self._data.min().item()

    def max(self) -> float:
        """Return the maximum value."""
        return self._data.max().item()

    # -- Functional methods -------------------------------------------------

    def apply(self, func: Callable[[Tensor], Any]) -> "Series":
        """Apply *func* to the underlying tensor.

        Parameters
        ----------
        func : callable
            A function ``Tensor -> Tensor`` or ``Tensor -> Series``.

        Returns
        -------
        Series
        """
        result = func(self._data)
        if isinstance(result, Series):
            return result
        return Series(result)

    def map(self, func: Callable) -> "Series":
        """Apply *func* element-wise.

        First attempts a vectorized call on the whole tensor; falls back to
        per-element application if the vectorized call fails.

        Parameters
        ----------
        func : callable

        Returns
        -------
        Series

        Raises
        ------
        TypeError
            If *func* is not callable.
        """
        if not callable(func):
            raise TypeError(f"map requires a callable, got {type(func).__name__}")
        try:
            result = func(self._data)
            if torch.is_tensor(result):
                return Series(result)
            result = torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype)
            return Series(result)
        except Exception:
            result = torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype)
            return Series(result)

    def agg(self, func: Union[str, Callable]) -> Any:
        """Aggregate the Series with a named function or callable.

        Parameters
        ----------
        func : str or callable
            One of ``'mean'``, ``'sum'``, ``'min'``, ``'max'``, ``'std'``,
            or a callable ``Tensor -> scalar``.

        Returns
        -------
        float or Tensor
        """
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
        raise ValueError(
            f"Unknown aggregation '{func}'. Supported: 'mean', 'sum', "
            "'min', 'max', 'std', or a callable."
        )

    def transform(self, func: Callable[[Tensor], Any]) -> "Series":
        """Apply *func* and return a same-shape Series.

        Parameters
        ----------
        func : callable
            A function ``Tensor -> Tensor`` (vectorized) or
            ``scalar -> scalar`` (per-element fallback).

        Returns
        -------
        Series
        """
        result = func(self._data)
        if torch.is_tensor(result):
            return Series(result)
        return Series(torch.tensor([func(x.item()) for x in self._data], dtype=self._data.dtype))

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Pipe ``self`` as the first argument to *func*.

        Equivalent to ``func(self, *args, **kwargs)``.
        """
        return func(self, *args, **kwargs)

    # -- Window methods -----------------------------------------------------

    def rolling(self, window: int) -> "Rolling":
        """Return a :class:`Rolling` window proxy.

        Parameters
        ----------
        window : int
            Window size (number of elements).  Must be >= 1.

        Returns
        -------
        Rolling
        """
        if window < 1:
            raise ValueError(f"rolling window must be >= 1, got {window}")
        return Rolling(self, window)

    def ewm(self, span: int) -> "EWM":
        """Return an :class:`EWM` (exponentially-weighted) proxy.

        Parameters
        ----------
        span : int
            Span for the EWM decay factor.  Must be >= 1.

        Returns
        -------
        EWM
        """
        if span < 1:
            raise ValueError(f"ewm span must be >= 1, got {span}")
        return EWM(self, span)

    def expanding(self) -> "Expanding":
        """Return an :class:`Expanding` window proxy."""
        return Expanding(self)

    # -- Value counts -------------------------------------------------------

    def value_counts(self) -> "Series":
        """Return counts of unique values.

        Returns
        -------
        Series
            Counts (int64) with an :class:`Index` containing the unique values.
        """
        unique_vals, counts = torch.unique(self._data, return_counts=True)
        return Series(counts.to(dtype=torch.long), index=Index({'values': unique_vals}))


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------

class DataFrame:
    """A 2-D tabular structure backed by ``Dict[str, Tensor]``.

    Drop-in replacement for ``pandas.DataFrame`` (subset).  Each column is a
    1-D :class:`torch.Tensor`.  The internal representation is compatible with
    ``torch.jit.script`` (``Dict[str, Tensor]``).

    Parameters
    ----------
    data : dict[str, Tensor]
        Column name → 1-D tensor mapping.

    Example
    -------
    >>> import xpandas as pd, torch
    >>> df = pd.DataFrame({
    ...     "price": torch.tensor([100.0, 200.0], dtype=torch.double),
    ...     "volume": torch.tensor([10.0, 20.0], dtype=torch.double),
    ... })
    >>> df.price.mean()
    150.0
    """

    def __init__(self, data: Dict[str, Tensor]) -> None:
        if not isinstance(data, dict):
            raise TypeError(
                f"DataFrame requires a dict of {{str: Tensor}}, got "
                f"{type(data).__name__}. Hint: pd.DataFrame({{'col': tensor}})"
            )
        self._data: Dict[str, Tensor] = dict(data)  # shallow copy
        self._index: Optional[Any] = None

    # -- Column access ------------------------------------------------------

    def __getattr__(self, name: str) -> "Series":
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._data:
            return Series(self._data[name])
        raise AttributeError(
            f"'DataFrame' has no column '{name}'. "
            f"Available columns: {list(self._data.keys())}"
        )

    def __getitem__(self, key: Any) -> Union["Series", "DataFrame"]:
        """Access columns by name, list of names, or boolean mask.

        Parameters
        ----------
        key : str, list[str], Series, or Tensor
            - ``str`` → single column as :class:`Series`
            - ``list[str]`` → sub-DataFrame
            - ``Series`` (bool) → row filter
            - ``Tensor`` (bool) → row filter

        Returns
        -------
        Series or DataFrame
        """
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(
                    f"Column '{key}' not found. "
                    f"Available: {list(self._data.keys())}"
                )
            return Series(self._data[key])
        elif isinstance(key, list):
            missing = [k for k in key if k not in self._data]
            if missing:
                raise KeyError(
                    f"Columns not found: {missing}. "
                    f"Available: {list(self._data.keys())}"
                )
            return DataFrame({k: self._data[k] for k in key})
        elif isinstance(key, Series):
            mask = key._data
            return DataFrame({k: v[mask] for k, v in self._data.items()})
        elif torch.is_tensor(key) and key.dtype == torch.bool:
            return DataFrame({k: v[key] for k, v in self._data.items()})
        else:
            raise KeyError(
                f"Unsupported key type: {type(key).__name__}. "
                "Use a str, list[str], boolean Series, or boolean Tensor."
            )

    def __setitem__(self, key: str, value: Tensor) -> None:
        """Set or create a column."""
        self._data[key] = value

    # -- GroupBy ------------------------------------------------------------

    def groupby(self, by: str) -> "GroupBy":
        """Group the DataFrame by column *by*.

        Parameters
        ----------
        by : str
            Column name to group by (must be int64 for enum-encoded keys).

        Returns
        -------
        GroupBy
        """
        if by not in self._data:
            raise KeyError(
                f"groupby column '{by}' not found. "
                f"Available: {list(self._data.keys())}"
            )
        return GroupBy(self, by)

    # -- Properties ---------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        """List of column names."""
        return list(self._data.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        """(n_rows, n_cols)."""
        if not self._data:
            return (0, 0)
        first = next(iter(self._data.values()))
        return (len(first), len(self._data))

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def __repr__(self) -> str:
        rows, cols = self.shape
        return f"DataFrame({rows} rows x {cols} cols, columns={self.columns})"

    @property
    def dtypes(self) -> Dict[str, torch.dtype]:
        """Column name → dtype mapping."""
        return {k: v.dtype for k, v in self._data.items()}

    @property
    def index(self) -> Any:
        """Row index (defaults to ``torch.arange(n_rows)``)."""
        if self._index is None:
            if not self._data:
                return torch.arange(0)
            n = len(next(iter(self._data.values())))
            return torch.arange(n)
        return self._index

    @index.setter
    def index(self, value: Any) -> None:
        self._index = value

    # -- Row selection / modification ---------------------------------------

    def head(self, n: int = 5) -> "DataFrame":
        """Return the first *n* rows."""
        return DataFrame({k: v[:n] for k, v in self._data.items()})

    def tail(self, n: int = 5) -> "DataFrame":
        """Return the last *n* rows."""
        return DataFrame({k: v[-n:] for k, v in self._data.items()})

    def drop(self, columns: Optional[Union[str, List[str]]] = None) -> "DataFrame":
        """Drop column(s).

        Parameters
        ----------
        columns : str or list[str], optional
            Column(s) to drop.

        Returns
        -------
        DataFrame
        """
        if columns is None:
            return DataFrame(dict(self._data))
        cols_to_drop = [columns] if isinstance(columns, str) else list(columns)
        return DataFrame({k: v for k, v in self._data.items() if k not in cols_to_drop})

    def rename(self, columns: Optional[Dict[str, str]] = None) -> "DataFrame":
        """Rename columns.

        Parameters
        ----------
        columns : dict[str, str], optional
            Mapping from old name → new name.

        Returns
        -------
        DataFrame
        """
        if columns is None:
            return DataFrame(dict(self._data))
        return DataFrame({columns.get(k, k): v for k, v in self._data.items()})

    @property
    def iloc(self) -> _iLocIndexer:
        """Integer-location based indexer."""
        return _iLocIndexer(self)

    @property
    def loc(self) -> _LocIndexer:
        """Label-based indexer."""
        return _LocIndexer(self)

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        """Sort rows by column *by*.

        Dispatches to ``xpandas::sort_by``.  Equivalent to
        ``df.sort_values(by)`` in pandas.

        Parameters
        ----------
        by : str
            Column to sort by.
        ascending : bool
            Sort order.

        Returns
        -------
        DataFrame
        """
        if by not in self._data:
            raise KeyError(
                f"sort column '{by}' not found. "
                f"Available: {list(self._data.keys())}"
            )
        sorted_data = torch.ops.xpandas.sort_by(self._data, by, ascending)
        return DataFrame(sorted_data)

    def reset_index(self, drop: bool = True) -> "DataFrame":
        """Reset the index to default integer range.

        Parameters
        ----------
        drop : bool
            If True (default), discard the old index.

        Returns
        -------
        DataFrame
        """
        new_df = DataFrame(dict(self._data))
        new_df._index = None
        return new_df

    def set_index(self, keys: str) -> "DataFrame":
        """Set column *keys* as the DataFrame index.

        The column is removed from the data and stored as the index.

        Parameters
        ----------
        keys : str
            Column name.

        Returns
        -------
        DataFrame
        """
        if keys not in self._data:
            raise KeyError(
                f"set_index column '{keys}' not found. "
                f"Available: {list(self._data.keys())}"
            )
        key_tensor = self._data[keys]
        new_data = {k: v for k, v in self._data.items() if k != keys}
        new_df = DataFrame(new_data)
        new_df._index = key_tensor
        return new_df

    # -- Functional methods -------------------------------------------------

    def apply(self, func: Callable, axis: int = 0) -> Union["DataFrame", "Series"]:
        """Apply *func* along an axis.

        Parameters
        ----------
        func : callable
            Function to apply.
        axis : int
            ``0`` = apply to each column, ``1`` = apply to each row.

        Returns
        -------
        DataFrame or Series
        """
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

    def applymap(self, func: Callable[[float], float]) -> "DataFrame":
        """Apply *func* element-wise to every cell.

        Parameters
        ----------
        func : callable
            Scalar → scalar function.

        Returns
        -------
        DataFrame
        """
        new_data = {}
        for k, v in self._data.items():
            new_data[k] = torch.tensor([func(x.item()) for x in v], dtype=v.dtype)
        return DataFrame(new_data)

    def agg(self, func: Union[str, Callable]) -> Dict[str, Any]:
        """Aggregate each column.

        Parameters
        ----------
        func : str or callable
            ``'mean'``, ``'sum'``, ``'min'``, ``'max'``, or a callable.

        Returns
        -------
        dict[str, float]
        """
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
        raise ValueError(
            f"Unknown aggregation '{func}'. "
            "Supported: 'mean', 'sum', 'min', 'max', or a callable."
        )

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Pipe ``self`` as the first argument to *func*."""
        return func(self, *args, **kwargs)

    # -- Merge / Join -------------------------------------------------------

    def merge(self, other: "DataFrame", on: str, how: str = 'inner') -> "DataFrame":
        """Inner-join two DataFrames on column *on*.

        Parameters
        ----------
        other : DataFrame
            Right DataFrame.
        on : str
            Join key column (must exist in both DataFrames).
        how : str
            Join type (currently only ``'inner'`` is supported).

        Returns
        -------
        DataFrame
        """
        if on not in self._data:
            raise KeyError(f"merge key '{on}' not in left DataFrame. Columns: {self.columns}")
        if on not in other._data:
            raise KeyError(f"merge key '{on}' not in right DataFrame. Columns: {other.columns}")
        left_key = self._data[on]
        right_key = other._data[on]
        right_index: Dict[Any, int] = {}
        for i, k in enumerate(right_key.tolist()):
            right_index[k] = i
        left_rows: List[int] = []
        right_rows: List[int] = []
        for i, k in enumerate(left_key.tolist()):
            if k in right_index:
                left_rows.append(i)
                right_rows.append(right_index[k])
        if not left_rows:
            cols = list(self._data.keys()) + [k for k in other._data.keys() if k != on]
            return DataFrame({c: torch.tensor([], dtype=self._data.get(c, list(other._data.values())[0]).dtype) for c in cols})
        left_idx = torch.tensor(left_rows, dtype=torch.long)
        right_idx = torch.tensor(right_rows, dtype=torch.long)
        result: Dict[str, Tensor] = {}
        for k, v in self._data.items():
            result[k] = v[left_idx]
        for k, v in other._data.items():
            if k != on:
                result[k] = v[right_idx]
        return DataFrame(result)

    def join(self, other: "DataFrame", on: Optional[str] = None, how: str = 'left') -> "DataFrame":
        """Join with *other* DataFrame.

        If *on* is given, delegates to :meth:`merge`.  Otherwise joins by
        row position.

        Parameters
        ----------
        other : DataFrame
        on : str, optional
        how : str

        Returns
        -------
        DataFrame
        """
        if on is not None:
            return self.merge(other, on=on, how=how)
        n = min(len(self), len(other))
        result = {k: v[:n] for k, v in self._data.items()}
        for k, v in other._data.items():
            if k not in result:
                result[k] = v[:n]
        return DataFrame(result)

    # -- Export methods -----------------------------------------------------

    def to_numpy(self) -> Any:
        """Convert to a numpy ndarray (columns stacked horizontally).

        Returns
        -------
        numpy.ndarray
        """
        import numpy as np
        arrays = [v.numpy() for v in self._data.values()]
        return np.column_stack(arrays) if arrays else np.array([])

    def to_dict(self, orient: str = 'dict') -> Any:
        """Convert to a Python dict.

        Parameters
        ----------
        orient : str
            ``'dict'`` or ``'list'`` → ``{col: [values]}``;
            ``'records'`` → ``[{col: val}, ...]``.

        Returns
        -------
        dict or list[dict]
        """
        if orient in ('dict', 'list'):
            return {k: v.tolist() for k, v in self._data.items()}
        elif orient == 'records':
            cols = list(self._data.keys())
            n = len(self)
            return [{c: self._data[c][i].item() for c in cols} for i in range(n)]
        raise ValueError(
            f"Unknown orient '{orient}'. Supported: 'dict', 'list', 'records'."
        )

    def describe(self) -> Dict[str, Dict[str, float]]:
        """Summary statistics for each column.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dict with keys ``count``, ``mean``, ``std``, ``min``,
            ``max`` per column.
        """
        result: Dict[str, Dict[str, float]] = {}
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

    def info(self) -> None:
        """Print a concise summary of the DataFrame."""
        print(f"DataFrame: {self.shape[0]} rows x {self.shape[1]} cols")
        for k, v in self._data.items():
            print(f"  {k}: {v.dtype}, {len(v)} non-null")


# ---------------------------------------------------------------------------
# GroupBy / GroupByColumn / Resampler
# ---------------------------------------------------------------------------

class GroupBy:
    """Grouping proxy returned by :meth:`DataFrame.groupby`.

    Use ``df.groupby('key')['value'].sum()`` to aggregate.  Every aggregation
    method dispatches to a ``torch.ops.xpandas.groupby_*`` C++ op.

    Example
    -------
    >>> gb = df.groupby("InstrumentID")
    >>> keys, sums = gb["price"].sum()  # returns (unique_keys, summed_values)
    """

    def __init__(self, df: DataFrame, by: str) -> None:
        self._df = df
        self._by = by

    def __getitem__(self, col: str) -> "GroupByColumn":
        """Select the value column to aggregate.

        Parameters
        ----------
        col : str
            Column name.

        Returns
        -------
        GroupByColumn
        """
        if col not in self._df._data:
            raise KeyError(
                f"GroupBy value column '{col}' not found. "
                f"Available: {self._df.columns}"
            )
        return GroupByColumn(self._df, self._by, col)


class GroupByColumn:
    """A GroupBy object with a specific value column selected.

    Provides aggregation methods (``sum``, ``mean``, ``count``, ``std``,
    ``min``, ``max``, ``first``, ``last``) and a :meth:`resample` method.

    Each aggregation returns a tuple ``(unique_keys: Tensor, values: Tensor)``.
    """

    def __init__(self, df: DataFrame, by: str, col: str) -> None:
        self._df = df
        self._by = by
        self._col = col

    def _key_value(self) -> Tuple[Tensor, Tensor]:
        return self._df._data[self._by], self._df._data[self._col]

    def sum(self) -> Tuple[Tensor, Tensor]:
        """Grouped sum.  Dispatches to ``xpandas::groupby_sum``."""
        return torch.ops.xpandas.groupby_sum(*self._key_value())

    def mean(self) -> Tuple[Tensor, Tensor]:
        """Grouped mean.  Dispatches to ``xpandas::groupby_mean``."""
        return torch.ops.xpandas.groupby_mean(*self._key_value())

    def count(self) -> Tuple[Tensor, Tensor]:
        """Grouped count.  Dispatches to ``xpandas::groupby_count``."""
        return torch.ops.xpandas.groupby_count(*self._key_value())

    def std(self) -> Tuple[Tensor, Tensor]:
        """Grouped standard deviation.  Dispatches to ``xpandas::groupby_std``."""
        return torch.ops.xpandas.groupby_std(*self._key_value())

    def min(self) -> Tuple[Tensor, Tensor]:
        """Grouped min.  Dispatches to ``xpandas::groupby_min``."""
        return torch.ops.xpandas.groupby_min(*self._key_value())

    def max(self) -> Tuple[Tensor, Tensor]:
        """Grouped max.  Dispatches to ``xpandas::groupby_max``."""
        return torch.ops.xpandas.groupby_max(*self._key_value())

    def first(self) -> Tuple[Tensor, Tensor]:
        """Grouped first value.  Dispatches to ``xpandas::groupby_first``."""
        return torch.ops.xpandas.groupby_first(*self._key_value())

    def last(self) -> Tuple[Tensor, Tensor]:
        """Grouped last value.  Dispatches to ``xpandas::groupby_last``."""
        return torch.ops.xpandas.groupby_last(*self._key_value())

    def resample(self, freq: str) -> "Resampler":
        """Resample with OHLC aggregation.

        Dispatches to ``xpandas::groupby_resample_ohlc`` which computes
        first/max/min/last per group in a single pass.

        Parameters
        ----------
        freq : str
            Frequency alias (unused in the C++ op; the groupby key already
            encodes the bucket).

        Returns
        -------
        Resampler
        """
        return Resampler(self._df, self._by, self._col, freq)


class Resampler:
    """OHLC resampler with cached computation.

    The underlying ``xpandas::groupby_resample_ohlc`` C++ op computes all
    four aggregations (Open, High, Low, Close) in a single pass.  Results
    are cached so that calling ``.first()``, ``.max()``, ``.min()``, and
    ``.last()`` in sequence does not re-compute.
    """

    def __init__(self, df: DataFrame, by: str, col: str, freq: str) -> None:
        self._key = df._data[by]
        self._value = df._data[col]
        self._cache: Optional[Tuple[Tensor, ...]] = None

    def _compute_ohlc(self) -> Tuple[Tensor, ...]:
        if self._cache is None:
            self._cache = torch.ops.xpandas.groupby_resample_ohlc(self._key, self._value)
        return self._cache

    def first(self) -> Series:
        """Open (first value per group)."""
        cache = self._compute_ohlc()
        return Series(cache[1], index=Index({'InstrumentID': cache[0]}))

    def max(self) -> Series:
        """High (max value per group)."""
        cache = self._compute_ohlc()
        return Series(cache[2], index=Index({'InstrumentID': cache[0]}))

    def min(self) -> Series:
        """Low (min value per group)."""
        cache = self._compute_ohlc()
        return Series(cache[3], index=Index({'InstrumentID': cache[0]}))

    def last(self) -> Series:
        """Close (last value per group)."""
        cache = self._compute_ohlc()
        return Series(cache[4], index=Index({'InstrumentID': cache[0]}))


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------

class Rolling:
    """Rolling-window proxy returned by :meth:`Series.rolling`.

    Each method dispatches to a ``torch.ops.xpandas.rolling_*`` C++ op.
    The first ``window - 1`` values are NaN.

    Example
    -------
    >>> s.rolling(5).mean()
    """

    def __init__(self, series: Series, window: int) -> None:
        self._series = series
        self._window = window

    def mean(self) -> Series:
        """Rolling mean.  Dispatches to ``xpandas::rolling_mean``."""
        return Series(torch.ops.xpandas.rolling_mean(self._series._data, self._window))

    def sum(self) -> Series:
        """Rolling sum.  Dispatches to ``xpandas::rolling_sum``."""
        return Series(torch.ops.xpandas.rolling_sum(self._series._data, self._window))

    def std(self) -> Series:
        """Rolling standard deviation.  Dispatches to ``xpandas::rolling_std``."""
        return Series(torch.ops.xpandas.rolling_std(self._series._data, self._window))

    def min(self) -> Series:
        """Rolling min (O(n) monotonic deque).  Dispatches to ``xpandas::rolling_min``."""
        return Series(torch.ops.xpandas.rolling_min(self._series._data, self._window))

    def max(self) -> Series:
        """Rolling max (O(n) monotonic deque).  Dispatches to ``xpandas::rolling_max``."""
        return Series(torch.ops.xpandas.rolling_max(self._series._data, self._window))


# ---------------------------------------------------------------------------
# EWM
# ---------------------------------------------------------------------------

class EWM:
    """Exponentially-weighted window proxy returned by :meth:`Series.ewm`.

    Dispatches to ``xpandas::ewm_mean``.

    Example
    -------
    >>> s.ewm(span=10).mean()
    """

    def __init__(self, series: Series, span: int) -> None:
        self._series = series
        self._span = span

    def mean(self) -> Series:
        """EWM mean (adjust=False).  Dispatches to ``xpandas::ewm_mean``."""
        return Series(torch.ops.xpandas.ewm_mean(self._series._data, self._span))


# ---------------------------------------------------------------------------
# Expanding
# ---------------------------------------------------------------------------

class Expanding:
    """Expanding-window proxy returned by :meth:`Series.expanding`.

    Example
    -------
    >>> s.expanding().sum()   # cumulative sum
    >>> s.expanding().mean()  # cumulative mean
    """

    def __init__(self, series: Series) -> None:
        self._series = series

    def sum(self) -> Series:
        """Expanding sum (cumulative).  Dispatches to ``xpandas::cumsum``."""
        return Series(torch.ops.xpandas.cumsum(self._series._data))

    def mean(self) -> Series:
        """Expanding mean: ``cumsum / [1, 2, ..., n]``."""
        data = self._series._data
        cs = torch.ops.xpandas.cumsum(data)
        counts = torch.arange(1, len(data) + 1, dtype=data.dtype, device=data.device)
        return Series(cs / counts)
