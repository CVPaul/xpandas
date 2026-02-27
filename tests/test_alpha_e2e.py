"""
tests/test_alpha_e2e.py -- End-to-end test: script Alpha, run on_bod + forward.

Run:
    pip install --no-build-isolation -e .
    pytest tests/test_alpha_e2e.py -v
"""

import torch
import xpandas  # noqa: F401

from examples.alpha_ts import Alpha
from examples.alpha_vwap import AlphaVWAP


# ============================================================
# Alpha (breakout signal) E2E tests
# ============================================================

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


# ============================================================
# AlphaVWAP (time-bucketed VWAP mean-reversion) E2E tests
# ============================================================

def _make_vwap_hist_data():
    """4 hours of tick data, 2 ticks per hour."""
    return {
        "timestamp": torch.tensor(
            [0, 1800, 3600, 5400, 7200, 9000, 10800, 12600], dtype=torch.long),
        "price": torch.tensor(
            [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0],
            dtype=torch.double),
        "volume": torch.tensor(
            [10.0, 20.0, 15.0, 25.0, 30.0, 10.0, 20.0, 20.0],
            dtype=torch.double),
    }


def test_vwap_script_and_run():
    """Script AlphaVWAP, call on_bod + forward, verify signals."""
    model = AlphaVWAP(window=3)
    scripted = torch.jit.script(model)

    hist = _make_vwap_hist_data()
    scripted.on_bod(0, hist)

    # VWAP per hour:
    #  H0: (100*10 + 102*20) / 30 = 3040/30 ≈ 101.333
    #  H1: (104*15 + 106*25) / 40 = 4210/40 = 105.25
    #  H2: (108*30 + 110*10) / 40 = 4340/40 = 108.5
    #  H3: (112*20 + 114*20) / 40 = 4520/40 = 113.0
    # Rolling mean (window=3): [NaN, NaN, 105.028, 108.917]
    # Last rolling avg ≈ 108.917

    # Price well above VWAP → short (-1)
    sig = scripted(0, {"price": torch.tensor([120.0], dtype=torch.double)})
    assert sig.item() == -1.0, f"Expected -1.0, got {sig.item()}"

    # Price well below VWAP → long (+1)
    sig = scripted(0, {"price": torch.tensor([100.0], dtype=torch.double)})
    assert sig.item() == 1.0, f"Expected 1.0, got {sig.item()}"


def test_vwap_flat_on_empty_history():
    """Forward before on_bod should return flat (0.0)."""
    model = AlphaVWAP(window=3)
    scripted = torch.jit.script(model)

    sig = scripted(0, {"price": torch.tensor([105.0], dtype=torch.double)})
    assert sig.item() == 0.0, f"Expected 0.0 (flat), got {sig.item()}"


def test_vwap_save_and_load(tmp_path):
    """Script, save, reload, verify AlphaVWAP produces same output."""
    model = AlphaVWAP(window=3)
    scripted = torch.jit.script(model)

    hist = _make_vwap_hist_data()
    scripted.on_bod(0, hist)

    path = str(tmp_path / "alpha_vwap_test.pt")
    scripted.save(path)

    loaded = torch.jit.load(path)
    # State was saved, so we can call forward directly
    sig_orig = scripted(0, {"price": torch.tensor([120.0], dtype=torch.double)})
    sig_load = loaded(0, {"price": torch.tensor([120.0], dtype=torch.double)})
    assert torch.equal(sig_orig, sig_load), (
        f"Mismatch after reload: {sig_orig} vs {sig_load}")


def test_vwap_multiple_instruments():
    """Test VWAP alpha with prices for multiple instruments at forward time."""
    model = AlphaVWAP(window=2)
    scripted = torch.jit.script(model)

    hist = _make_vwap_hist_data()
    scripted.on_bod(0, hist)

    # Multiple prices: one above, one below, one equal to ref VWAP
    ref = scripted.rolling_avg[-1].item()
    tick = {
        "price": torch.tensor([ref + 10.0, ref - 10.0, ref],
                              dtype=torch.double),
    }
    sig = scripted(0, tick)
    expected = torch.tensor([-1.0, 1.0, 0.0], dtype=torch.double)
    assert torch.equal(sig, expected), f"Expected {expected}, got {sig}"
