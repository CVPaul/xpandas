#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
alpha_ts.py -- TorchScript-compatible rewrite of alpha.py.

Data model (matches the C++ engine contract):
  - A "DataFrame" is  Dict[str, Tensor]
  - String columns (e.g. InstrumentID) are enum-encoded to int64 tensors
  - Numeric columns (e.g. price) are float64 tensors
  - All columns have the same length (= number of rows)

Usage:
  Python side:
      import xpandas                               # loads custom ops
      from alpha_ts import Alpha
      m = torch.jit.script(Alpha())
      m.save("alpha.pt")

  C++ side:
      auto m = torch::jit::load("alpha.pt");       # needs libxpandas_ops.so
      m.get_method("on_bod")({ts, data});
      auto signal = m.get_method("forward")({ts, data});
"""

import torch
from torch import Tensor
from typing import Dict, Tuple


class Alpha(torch.nn.Module):
    """Breakout / momentum signal -- TorchScript-compatible.

    State:
        hist_inst  : Tensor (int64)   – instrument ids
        hist_high  : Tensor (float64) – daily high per instrument
        hist_low   : Tensor (float64) – daily low  per instrument
    """

    def __init__(self):
        super().__init__()
        # Persisted state across on_bod → forward calls.
        # Register as buffers so they travel with save/load.
        self.register_buffer('hist_inst', torch.empty(0, dtype=torch.long))
        self.register_buffer('hist_high', torch.empty(0, dtype=torch.double))
        self.register_buffer('hist_low',  torch.empty(0, dtype=torch.double))

    @torch.jit.export
    def on_bod(self, timestamp: int, data: Dict[str, Tensor]) -> None:
        """Called at the beginning of the day with historical tick data.

        Computes per-instrument OHLC and stores high/low for the day.

        Args:
            timestamp: epoch seconds (unused in logic, kept for interface)
            data:      {"InstrumentID": int64 Tensor, "price": float64 Tensor}
        """
        key   = data["InstrumentID"]
        value = data["price"]

        # Custom op: groupby key, compute OHLC on value
        inst, _open, high, low, _close = torch.ops.xpandas.groupby_resample_ohlc(key, value)

        self.hist_inst = inst
        self.hist_high = high
        self.hist_low  = low

    def forward(self, timestamp: int, data: Dict[str, Tensor]) -> Tensor:
        """Called every interval (1s). Returns per-instrument signal.

        Returns:
            Tensor of float64:  +1.0 (long), -1.0 (short), 0.0 (flat)
        """
        price = data["price"]

        # Custom op: fused breakout signal
        signal = torch.ops.xpandas.breakout_signal(price, self.hist_high, self.hist_low)
        return signal
