import torch
import xpandas as pd

class Alpha(torch.nn.Module):
    """Adapted from alpha.py — adds super().__init__() for Module compatibility."""
    def __init__(self, source: dict):
        super().__init__()
        for k, v in source.items():
            setattr(self, k, v)
        self.freq = source.get('freq', '1s')

    def on_bod(self, timestamp: int, data: pd.DataFrame):
        gs = data.groupby('InstrumentID')
        o = gs['price'].resample('1D').first()
        h = gs['price'].resample('1D').max()
        l = gs['price'].resample('1D').min()
        c = gs['price'].resample('1D').last()
        self.hist = pd.concat([
            pd.DataFrame({'inst': o.index.get_level_values('InstrumentID'),
                          'open': o.values, 'high': h.values,
                          'low': l.values, 'close': c.values})
        ])

    def forward(self, timestamp: int, data: pd.DataFrame):
        long_ = data.price > self.hist.high
        short = data.price < self.hist.low
        return long_.astype(float) - short.astype(float)


def _make_bod_data():
    """2 instruments, 3 prices each: inst0=[100,105,102], inst1=[200,198,210]"""
    return pd.DataFrame({
        'InstrumentID': torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        'price': torch.tensor([100., 105., 102., 200., 198., 210.], dtype=torch.double)
    })


def test_alpha_e2e_breakout_signal():
    """Price above high → +1, price below low → -1."""
    m = Alpha({'freq': '1D'})
    m.on_bod(0, _make_bod_data())
    # inst0: open=100, high=105, low=100, close=102 → fwd price 106 > high → +1
    # inst1: open=200, high=210, low=198, close=210 → fwd price 195 < low → -1
    fwd = pd.DataFrame({'price': torch.tensor([106., 195.], dtype=torch.double)})
    sig = m.forward(0, fwd)
    sig_t = sig.values if hasattr(sig, 'values') else sig
    expected = torch.tensor([1.0, -1.0], dtype=torch.double)
    assert torch.equal(sig_t, expected), f'Expected {expected}, got {sig}'


def test_alpha_e2e_no_breakout():
    """Prices within OHLC range → signal = [0.0, 0.0]."""
    m = Alpha({'freq': '1D'})
    m.on_bod(0, _make_bod_data())
    # inst0: high=105, low=100 → 102 is inside → 0
    # inst1: high=210, low=198 → 200 is inside → 0
    fwd = pd.DataFrame({'price': torch.tensor([102., 200.], dtype=torch.double)})
    sig = m.forward(0, fwd)
    sig_t = sig.values if hasattr(sig, 'values') else sig
    expected = torch.tensor([0.0, 0.0], dtype=torch.double)
    assert torch.equal(sig_t, expected), f'Expected {expected}, got {sig}'


def test_alpha_e2e_return_type():
    """forward() returns Series or Tensor backed by float data."""
    m = Alpha({'freq': '1D'})
    m.on_bod(0, _make_bod_data())
    fwd = pd.DataFrame({'price': torch.tensor([106., 195.], dtype=torch.double)})
    sig = m.forward(0, fwd)
    sig_t = sig.values if hasattr(sig, 'values') else sig
    assert isinstance(sig_t, torch.Tensor), f'Expected Tensor (or Series wrapping Tensor), got {type(sig)}'


def test_alpha_e2e_source_config():
    """Alpha({'freq': '1D'}) correctly sets freq attribute from source dict."""
    m = Alpha({'freq': '1D'})
    assert m.freq == '1D'


def test_alpha_e2e_single_instrument():
    """Works correctly with only 1 instrument."""
    m = Alpha({})
    bod = pd.DataFrame({
        'InstrumentID': torch.tensor([0, 0, 0], dtype=torch.long),
        'price': torch.tensor([50., 55., 52.], dtype=torch.double)
    })
    m.on_bod(0, bod)
    # high=55, low=50 → 60 > high → +1
    fwd = pd.DataFrame({'price': torch.tensor([60.], dtype=torch.double)})
    sig = m.forward(0, fwd)
    sig_t = sig.values if hasattr(sig, 'values') else sig
    assert sig_t.tolist() == [1.0], f'Expected [1.0], got {sig_t.tolist()}'
