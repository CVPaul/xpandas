"""
xpandas/ops_meta.py -- FakeTensor (meta) kernels for xpandas custom ops.

These tell torch.compile what shape/dtype the outputs will have, without
actually running the computation. Not required for torch.jit.script, but
needed for torch.compile / torch.export support.
"""

import torch
from torch import Tensor
from typing import Dict, Tuple


@torch.library.register_fake("xpandas::groupby_resample_ohlc")
def groupby_resample_ohlc_fake(
    key: Tensor, value: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    n = key.size(0)
    return (
        torch.empty(n, dtype=torch.long,   device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
    )


@torch.library.register_fake("xpandas::compare_gt")
def compare_gt_fake(a: Tensor, b: Tensor) -> Tensor:
    return torch.empty_like(a, dtype=torch.bool)


@torch.library.register_fake("xpandas::compare_lt")
def compare_lt_fake(a: Tensor, b: Tensor) -> Tensor:
    return torch.empty_like(a, dtype=torch.bool)


@torch.library.register_fake("xpandas::bool_to_float")
def bool_to_float_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


@torch.library.register_fake("xpandas::lookup")
def lookup_fake(table: Dict[str, Tensor], key: str) -> Tensor:
    return torch.empty(0, dtype=torch.double)


@torch.library.register_fake("xpandas::breakout_signal")
def breakout_signal_fake(price: Tensor, high: Tensor, low: Tensor) -> Tensor:
    return torch.empty_like(price, dtype=torch.double)


@torch.library.register_fake("xpandas::rank")
def rank_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


@torch.library.register_fake("xpandas::to_datetime")
def to_datetime_fake(epochs: Tensor, unit: str) -> Tensor:
    return torch.empty_like(epochs, dtype=torch.long)


@torch.library.register_fake("xpandas::dt_floor")
def dt_floor_fake(dt_ns: Tensor, interval_ns: int) -> Tensor:
    return torch.empty_like(dt_ns, dtype=torch.long)


# --- groupby aggregations ---

@torch.library.register_fake("xpandas::groupby_sum")
def groupby_sum_fake(key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    n = key.size(0)
    return (
        torch.empty(n, dtype=torch.long, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
    )


@torch.library.register_fake("xpandas::groupby_mean")
def groupby_mean_fake(key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    n = key.size(0)
    return (
        torch.empty(n, dtype=torch.long, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
    )


@torch.library.register_fake("xpandas::groupby_count")
def groupby_count_fake(key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    n = key.size(0)
    return (
        torch.empty(n, dtype=torch.long, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
    )


@torch.library.register_fake("xpandas::groupby_std")
def groupby_std_fake(key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
    n = key.size(0)
    return (
        torch.empty(n, dtype=torch.long, device=key.device),
        torch.empty(n, dtype=torch.double, device=key.device),
    )


# --- rolling window ---

@torch.library.register_fake("xpandas::rolling_sum")
def rolling_sum_fake(x: Tensor, window: int) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


@torch.library.register_fake("xpandas::rolling_mean")
def rolling_mean_fake(x: Tensor, window: int) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


@torch.library.register_fake("xpandas::rolling_std")
def rolling_std_fake(x: Tensor, window: int) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


# --- shift ---

@torch.library.register_fake("xpandas::shift")
def shift_fake(x: Tensor, periods: int) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


# --- fillna ---

@torch.library.register_fake("xpandas::fillna")
def fillna_fake(x: Tensor, fill_value: float) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


# --- conditional ---

@torch.library.register_fake("xpandas::where_")
def where_fake(cond: Tensor, x: Tensor, other: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.register_fake("xpandas::masked_fill")
def masked_fill_fake(x: Tensor, mask: Tensor, fill_value: float) -> Tensor:
    return torch.empty_like(x)


# --- pct_change ---

@torch.library.register_fake("xpandas::pct_change")
def pct_change_fake(x: Tensor, periods: int) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


# --- cumulative ---

@torch.library.register_fake("xpandas::cumsum")
def cumsum_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


@torch.library.register_fake("xpandas::cumprod")
def cumprod_fake(x: Tensor) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)


# --- clip ---

@torch.library.register_fake("xpandas::clip")
def clip_fake(x: Tensor, lower: float, upper: float) -> Tensor:
    return torch.empty_like(x, dtype=torch.double)
