#!/usr/bin/env python
#-*- coding:utf-8 -*-


import torch
import pandas as pd


class Alpha(torch.nn.Module):
    
    def __init__(self, source: dict):
        for k, v in source.items():
            setattr(self, k, v)
        self.freq = '1s'
    
    # call at the begin of the day
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
    
    # call at the end of each interval(frequency=self.freq)
    def forward(self, timestamp: int, data: pd.DataFrame):
        long_ = data.price > self.hist.high
        short = data.price < self.hist.low
        return long_.astype(float) - short.astype(float)


# The outer C++ engine fetches market data and calls on_bod / forward.
# Data contains only string and double types.
# Goal: replace `import pandas as pd` with `import xpandas as pd` to record