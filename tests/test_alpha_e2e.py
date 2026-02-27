"""
tests/test_alpha_e2e.py -- End-to-end test: script Alpha, run on_bod + forward.

Run:
    pip install -e .
    pytest tests/test_alpha_e2e.py -v
"""

import torch
import xpandas  # noqa: F401

from examples.alpha_ts import Alpha


def test_script_and_run():
    """Script the Alpha model, call on_bod + forward, verify signal."""
    model = Alpha()
    scripted = torch.jit.script(model)

    # Historical data
    hist_data = {
        "InstrumentID": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "price": torch.tensor([100.0, 105.0, 102.0, 200.0, 198.0, 210.0],
                              dtype=torch.double),
    }
    scripted.on_bod(1700000000, hist_data)

    # Current tick
    tick_data = {
        "price": torch.tensor([106.0, 195.0], dtype=torch.double),
    }
    signal = scripted(1700000000, tick_data)

    expected = torch.tensor([1.0, -1.0], dtype=torch.double)
    assert torch.equal(signal, expected), f"Expected {expected}, got {signal}"


def test_save_and_load(tmp_path):
    """Script, save, reload, and verify the model produces same output."""
    model = Alpha()
    scripted = torch.jit.script(model)

    path = str(tmp_path / "alpha_test.pt")
    scripted.save(path)

    loaded = torch.jit.load(path)

    hist_data = {
        "InstrumentID": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        "price": torch.tensor([100.0, 105.0, 102.0, 200.0, 198.0, 210.0],
                              dtype=torch.double),
    }
    loaded.on_bod(1700000000, hist_data)

    tick_data = {
        "price": torch.tensor([106.0, 195.0], dtype=torch.double),
    }
    signal = loaded(1700000000, tick_data)

    expected = torch.tensor([1.0, -1.0], dtype=torch.double)
    assert torch.equal(signal, expected)
