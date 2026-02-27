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
