"""
xpandas -- drop-in *subset* of the pandas API backed by torch.ops.xpandas.*
custom operators.  Because every operation is a registered torch op, a
torch.nn.Module that uses xpandas can be compiled with torch.jit.script and
later loaded in pure C++ with torch::jit::load().

Usage:
    import xpandas as pd          # instead of  import pandas as pd
"""

from pathlib import Path
import torch

# --------------------------------------------------------------------------
# Load the compiled C++ extension that registers the xpandas:: ops.
# We look for a _C*.so next to this __init__.py (built by setup.py).
# --------------------------------------------------------------------------
_so_files = list(Path(__file__).parent.glob("_C*"))
if _so_files:
    torch.ops.load_library(str(_so_files[0]))

from .wrappers import DataFrame, Series, Index, GroupBy, Rolling, EWM, Expanding


# --------------------------------------------------------------------------
# Public API -- thin wrappers that TorchScript can understand
# --------------------------------------------------------------------------



def concat(frames, axis=0):
    """Concatenate DataFrames along axis.
    
    axis=0: vertical stack (same columns)
    axis=1: horizontal merge (different columns)
    """
    if isinstance(frames, list) and len(frames) == 1:
        return frames[0]
    if not frames:
        from .wrappers import DataFrame
        return DataFrame({})
    if axis == 1:
        # Horizontal: merge all columns
        result = {}
        for f in frames:
            result.update(f._data)
        from .wrappers import DataFrame
        return DataFrame(result)
    else:
        # Vertical: stack same columns
        cols = list(frames[0]._data.keys())
        result = {}
        for c in cols:
            result[c] = torch.cat([f._data[c] for f in frames])
        from .wrappers import DataFrame
        return DataFrame(result)


# --------------------------------------------------------------------------
# Datetime constants (nanoseconds) -- mirrors pandas offset aliases
# --------------------------------------------------------------------------
NS_PER_SECOND = 1_000_000_000
NS_PER_MINUTE = 60 * NS_PER_SECOND
NS_PER_HOUR   = 60 * NS_PER_MINUTE
NS_PER_DAY    = 24 * NS_PER_HOUR

_FREQ_TO_NS = {
    "1s":   NS_PER_SECOND,
    "1T":   NS_PER_MINUTE,   # pandas alias
    "1min": NS_PER_MINUTE,
    "1h":   NS_PER_HOUR,
    "1H":   NS_PER_HOUR,     # pandas alias
    "1D":   NS_PER_DAY,
    "1d":   NS_PER_DAY,
}


def to_datetime(series, unit: str = "s"):
    """Convert epoch timestamps to nanosecond epoch (int64 Tensor).

    Drop-in replacement for ``pd.to_datetime(series, unit=unit)``.

    Args:
        series: Tensor of epoch timestamps (int64 or float64).
        unit:   One of 's', 'ms', 'us', 'ns'.

    Returns:
        int64 Tensor of nanosecond-precision epoch timestamps.
    """
    return torch.ops.xpandas.to_datetime(series, unit)


def dt_floor(dt_ns, freq: str):
    """Floor nanosecond-epoch timestamps to the given frequency.

    Drop-in replacement for ``series.dt.floor(freq)``.

    Args:
        dt_ns: int64 Tensor of nanosecond epoch timestamps
               (as returned by ``to_datetime``).
        freq:  Frequency string, e.g. '1D', '1h', '1min', '1s'.

    Returns:
        int64 Tensor of floored timestamps (can be used as groupby key).
    """
    if freq in _FREQ_TO_NS:
        interval_ns = _FREQ_TO_NS[freq]
    else:
        raise ValueError(
            f"Unsupported freq '{freq}'. Supported: {list(_FREQ_TO_NS.keys())}"
        )
    return torch.ops.xpandas.dt_floor(dt_ns, interval_ns)


__all__ = ["DataFrame", "Series", "Index", "GroupBy", "Rolling", "EWM", "Expanding", "concat", "to_datetime", "dt_floor"]
