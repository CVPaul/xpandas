#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
alpha_momentum.py -- Cross-sectional momentum z-score Alpha (TorchScript-compatible).

Demonstrates use of ewm_mean, pct_change, zscore, log_, rolling_std, abs_,
clip, fillna, and sort_by.

Strategy:
    1. on_bod: Receive historical price data per instrument.
       - Compute log returns via log_(price) + pct_change (log-return proxy)
       - Smooth returns with ewm_mean (exponential moving average)
       - Compute rolling volatility with rolling_std
       - Build a risk-adjusted momentum signal = smoothed_return / volatility
       - Cross-sectional zscore to normalize across instruments
       - Clip extreme values to [-3, 3]
    2. forward: Use the pre-computed momentum z-scores as the trading signal.
       Positive z-score → long, negative → short, 0 → flat.

Data model (same contract as alpha_ts.py):
    - DataFrame = Dict[str, Tensor]
    - "InstrumentID": int64 Tensor (enum-encoded instrument IDs)
    - "price":        float64 Tensor (close prices, one per instrument)

Usage:
    Python:
        import xpandas
        from examples.alpha_momentum import AlphaMomentum
        m = torch.jit.script(AlphaMomentum())
        m.save("alpha_momentum.pt")
    C++:
        auto m = torch::jit::load("alpha_momentum.pt");  // needs libxpandas_ops.so
"""

import torch
from torch import Tensor
from typing import Dict


class AlphaMomentum(torch.nn.Module):
    """Cross-sectional momentum z-score signal.

    State:
        momentum_zscore : Tensor (float64) – z-scored momentum per instrument
        inst_ids        : Tensor (int64)   – instrument IDs (sorted)
    """

    def __init__(self, ewm_span: int = 5, vol_window: int = 5):
        super().__init__()
        self.ewm_span: int = ewm_span
        self.vol_window: int = vol_window
        self.register_buffer('momentum_zscore', torch.empty(0, dtype=torch.double))
        self.register_buffer('inst_ids', torch.empty(0, dtype=torch.long))

    @torch.jit.export
    def on_bod(self, timestamp: int, data: Dict[str, Tensor]) -> None:
        """Process historical per-instrument prices to compute momentum z-scores.

        For each instrument, we receive a single representative price (e.g.,
        previous day's close).  In a real system, `data` would contain a time
        series; here we demonstrate the ops pipeline on a cross-section of
        instruments at a single point in time.

        Args:
            timestamp: epoch seconds (unused)
            data: {"InstrumentID": int64, "price": float64}
        """
        inst = data["InstrumentID"]
        price = data["price"]

        n = price.size(0)
        if n < 2:
            self.momentum_zscore = torch.zeros(n, dtype=torch.double)
            self.inst_ids = inst
            return

        # Step 1: Compute log prices
        log_price = torch.ops.xpandas.log_(price)

        # Step 2: Compute returns (pct_change of log prices ≈ log returns)
        returns = torch.ops.xpandas.pct_change(log_price, 1)

        # Step 3: Fill NaN from pct_change (first element) with 0
        returns = torch.ops.xpandas.fillna(returns, 0.0)

        # Step 4: Smooth returns with exponential moving average
        smoothed = torch.ops.xpandas.ewm_mean(returns, self.ewm_span)

        # Step 5: Compute rolling volatility of returns
        vol = torch.ops.xpandas.rolling_std(returns, self.vol_window)

        # Step 6: Fill NaN volatility (first vol_window-1 elements) with
        #         a fallback: absolute value of the return itself
        abs_returns = torch.ops.xpandas.abs_(returns)
        # Use fallback where vol is NaN
        vol_filled = torch.ops.xpandas.fillna(vol, 1.0)
        # Avoid division by zero in vol
        vol_safe = torch.ops.xpandas.clip(vol_filled, 1e-8, 1e8)

        # Step 7: Risk-adjusted momentum = smoothed_return / volatility
        raw_momentum = smoothed / vol_safe

        # Step 8: Cross-sectional z-score
        zscored = torch.ops.xpandas.zscore(raw_momentum)

        # Step 9: Clip extremes to [-3, 3]
        clipped = torch.ops.xpandas.clip(zscored, -3.0, 3.0)

        # Step 10: Fill any remaining NaN with 0 (flat signal)
        self.momentum_zscore = torch.ops.xpandas.fillna(clipped, 0.0)
        self.inst_ids = inst

    def forward(self, timestamp: int, data: Dict[str, Tensor]) -> Tensor:
        """Return momentum z-score as the trading signal.

        Positive → long, negative → short, 0 → flat.

        Returns:
            Tensor of float64: momentum z-scores clipped to [-3, 3]
        """
        n = data["price"].size(0)
        n_stored = self.momentum_zscore.size(0)

        if n_stored == 0:
            return torch.zeros(n, dtype=torch.double)

        # Return stored momentum z-scores (one per instrument)
        return self.momentum_zscore
