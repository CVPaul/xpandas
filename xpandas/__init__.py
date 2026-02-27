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


# --------------------------------------------------------------------------
# Public API -- thin wrappers that TorchScript can understand
# --------------------------------------------------------------------------

def DataFrame(d: dict):
    """Construct a "DataFrame" from a dict of column-name -> Tensor.

    Returns Dict[str, Tensor] directly -- TorchScript-friendly.
    """
    return d


def concat(frames):
    """Placeholder concat -- for the OHLC case we construct the frame
    directly inside on_bod, so this is identity."""
    if isinstance(frames, list) and len(frames) == 1:
        return frames[0]
    raise NotImplementedError("concat of >1 frames not yet supported")


__all__ = ["DataFrame", "concat"]
