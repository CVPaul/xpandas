#!/usr/bin/env python
"""
trace_and_save.py -- Script the Alpha model and save it as TorchScript.

Prerequisites:
    pip install -e .          # builds the xpandas C++ extension

Usage:
    python trace_and_save.py  # produces alpha.pt

Then on the C++ side:
    ./alpha_infer alpha.pt libxpandas_ops.so
"""

import torch
import xpandas  # noqa: F401 -- loads the custom ops .so

from examples.alpha_ts import Alpha


def main():
    # ------------------------------------------------------------------
    # 1. Script the model
    # ------------------------------------------------------------------
    model = Alpha()
    scripted = torch.jit.script(model)
    print("TorchScript IR for Alpha:")
    print(scripted.graph)
    print()

    # ------------------------------------------------------------------
    # 2. Quick functional test in Python
    # ------------------------------------------------------------------
    # Simulate historical tick data for on_bod
    hist_data = {
        "InstrumentID": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "price": torch.tensor([100.0, 105.0, 102.0, 200.0, 198.0, 210.0],
                              dtype=torch.double),
    }
    scripted.on_bod(1700000000, hist_data)
    print("After on_bod:")
    print(f"  hist_inst = {scripted.hist_inst}")
    print(f"  hist_high = {scripted.hist_high}")
    print(f"  hist_low  = {scripted.hist_low}")
    print()

    # Forward: prices that trigger signals
    tick_data = {
        "price": torch.tensor([106.0, 195.0], dtype=torch.double),
    }
    signal = scripted(1700000000, tick_data)
    print(f"Signal = {signal}")
    print(f"Expected: [ 1., -1.]  (inst0 breaks above 105, inst1 breaks below 198)")
    print()

    # Verify
    expected = torch.tensor([1.0, -1.0], dtype=torch.double)
    assert torch.equal(signal, expected), f"MISMATCH: got {signal}"
    print("Functional test PASSED.")

    # ------------------------------------------------------------------
    # 3. Save
    # ------------------------------------------------------------------
    out_path = "alpha.pt"
    scripted.save(out_path)
    print(f"\nModel saved to {out_path}")
    print("To run in C++:")
    print(f"  ./alpha_infer {out_path} <path-to-libxpandas_ops.so>")


if __name__ == "__main__":
    main()
