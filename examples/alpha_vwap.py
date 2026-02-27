#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
alpha_vwap.py -- Time-bucketed VWAP mean-reversion Alpha (TorchScript-compatible).

Demonstrates use of dt_floor-based groupby for time bucketing, along with
rolling window and shift ops.

Strategy:
    1. on_bod: Receive historical tick data with timestamps.  Convert
       timestamps to nanoseconds, floor to hourly buckets, then compute
       per-bucket VWAP (volume-weighted average price).
    2. forward: Compare the current price to a rolling mean of recent
       VWAPs.  If price > rolling_mean → short (mean-revert down);
       if price < rolling_mean → long (mean-revert up).  Otherwise flat.

Data model (same contract as alpha_ts.py):
    - DataFrame = Dict[str, Tensor]
    - "timestamp": int64 Tensor of epoch seconds
    - "price":     float64 Tensor
    - "volume":    float64 Tensor

Usage:
    Python:
        import xpandas
        from examples.alpha_vwap import AlphaVWAP
        m = torch.jit.script(AlphaVWAP())
        m.save("alpha_vwap.pt")
    C++:
        auto m = torch::jit::load("alpha_vwap.pt");  // needs libxpandas_ops.so
"""

import torch
from torch import Tensor
from typing import Dict


class AlphaVWAP(torch.nn.Module):
    """Mean-reversion signal based on time-bucketed VWAP.

    State:
        vwap_series : Tensor (float64) – hourly VWAP values from on_bod
        rolling_avg : Tensor (float64) – rolling mean of recent VWAPs
    """

    def __init__(self, window: int = 3):
        super().__init__()
        self.window: int = window
        self.register_buffer('vwap_series', torch.empty(0, dtype=torch.double))
        self.register_buffer('rolling_avg', torch.empty(0, dtype=torch.double))

    @torch.jit.export
    def on_bod(self, timestamp: int, data: Dict[str, Tensor]) -> None:
        """Process historical tick data: bucket by hour, compute VWAP per bucket.

        Args:
            timestamp: epoch seconds (unused in logic)
            data: {"timestamp": int64, "price": float64, "volume": float64}
        """
        ts_epochs = data["timestamp"]   # epoch seconds, int64
        price     = data["price"]       # float64
        volume    = data["volume"]      # float64

        # Step 1: Convert epoch seconds → nanosecond timestamps
        ts_ns = torch.ops.xpandas.to_datetime(ts_epochs, "s")

        # Step 2: Floor to 1-hour buckets (key for groupby)
        #   1 hour = 3_600_000_000_000 nanoseconds
        bucket_key = torch.ops.xpandas.dt_floor(ts_ns, 3600000000000)

        # Step 3: Compute per-bucket sum(price * volume) and sum(volume)
        #   VWAP = sum(price * volume) / sum(volume)
        pv = price * volume  # element-wise price * volume

        _keys_pv, sum_pv = torch.ops.xpandas.groupby_sum(bucket_key, pv)
        _keys_v, sum_v   = torch.ops.xpandas.groupby_sum(bucket_key, volume)

        # Avoid division by zero
        vwap = sum_pv / (sum_v + 1e-12)
        self.vwap_series = vwap

        # Step 4: Compute rolling mean of VWAP series
        self.rolling_avg = torch.ops.xpandas.rolling_mean(vwap, self.window)

    def forward(self, timestamp: int, data: Dict[str, Tensor]) -> Tensor:
        """Called every interval. Compare current price to latest rolling VWAP.

        Returns:
            Tensor of float64: +1.0 (long/undervalued), -1.0 (short/overvalued),
                                0.0 (flat/insufficient history)
        """
        price = data["price"]  # float64, one per instrument

        # Use the last rolling average as the reference VWAP level.
        # rolling_avg may have NaN at the start (< window elements), so
        # use the last valid value.
        n_vwap = self.rolling_avg.size(0)

        if n_vwap == 0:
            # No history yet — return flat
            return torch.zeros_like(price)

        # Grab the last rolling VWAP value
        ref_vwap = self.rolling_avg[n_vwap - 1]

        # Mean-reversion: price above VWAP → short, below → long
        long_signal  = torch.ops.xpandas.compare_lt(price,
                           ref_vwap.expand(price.size(0)))
        short_signal = torch.ops.xpandas.compare_gt(price,
                           ref_vwap.expand(price.size(0)))

        long_f  = torch.ops.xpandas.bool_to_float(long_signal)
        short_f = torch.ops.xpandas.bool_to_float(short_signal)

        return long_f - short_f
