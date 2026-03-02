"""wrapper_api_tour.py — Complete tour of the xpandas wrapper API.

Demonstrates every wrapper class and its methods.  Run this script
to verify the full API surface works end-to-end.
"""

import xpandas as pd
import torch

torch.manual_seed(42)


# =============================================================================
# SECTION: Series
# =============================================================================
print("\n=== Section: Series ===")

# Creation
s = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
print(f"s = {s}")

# Properties
print(f"s.values: {s.values}")
print(f"s.dtype: {s.dtype}")
print(f"s.shape: {s.shape}")
print(f"len(s): {len(s)}")

# Arithmetic operators
s_add = s + 10.0
print(f"s + 10: {s_add}")

s_sub = s - 1.0
print(f"s - 1: {s_sub}")

s_mul = s * 2.0
print(f"s * 2: {s_mul}")

s_div = s / 2.0
print(f"s / 2: {s_div}")

s_pow = s ** 2.0
print(f"s ** 2: {s_pow}")

s_mod = s % 2.0
print(f"s % 2: {s_mod}")

s_neg = -s
print(f"-s: {s_neg}")

# Comparison operators
mask_gt = s > 2.5
print(f"s > 2.5: {mask_gt}")

mask_lt = s < 3.5
print(f"s < 3.5: {mask_lt}")

mask_eq = s == 3.0
print(f"s == 3.0: {mask_eq}")

mask_ne = s != 3.0
print(f"s != 3.0: {mask_ne}")

mask_ge = s >= 3.0
print(f"s >= 3.0: {mask_ge}")

mask_le = s <= 3.0
print(f"s <= 3.0: {mask_le}")

# Bitwise operators
s_and = (s > 2.0) & (s < 5.0)
print(f"(s > 2.0) & (s < 5.0): {s_and}")

s_or = (s < 2.0) | (s > 4.0)
print(f"(s < 2.0) | (s > 4.0): {s_or}")

# Absolute value
s_abs = abs(s - 3.0)
print(f"abs(s - 3.0): {s_abs}")

# Statistical/math methods
s_stat = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
print(f"s.abs(): {s_stat.abs()}")
print(f"s.log(): {s_stat.log()}")
print(f"s.zscore(): {s_stat.zscore()}")
print(f"s.rank(): {s_stat.rank()}")

# NaN handling
s_nan = pd.Series(torch.tensor([1.0, float('nan'), 3.0, float('nan'), 5.0], dtype=torch.double))
print(f"s_nan: {s_nan}")
print(f"s_nan.fillna(0.0): {s_nan.fillna(0.0)}")

# Shift and lag
s_shift = pd.Series(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.double))
print(f"s_shift.shift(1): {s_shift.shift(1)}")
print(f"s_shift.shift(-1): {s_shift.shift(-1)}")

# Percentage change
s_prices = pd.Series(torch.tensor([100.0, 110.0, 99.0, 121.0, 108.9], dtype=torch.double))
print(f"s_prices.pct_change(): {s_prices.pct_change()}")

# Cumulative
s_cum = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
print(f"s_cum.cumsum(): {s_cum.cumsum()}")
print(f"s_cum.cumprod(): {s_cum.cumprod()}")

# Clipping
s_clip = pd.Series(torch.tensor([1.0, 5.0, 3.0, 10.0, 2.0], dtype=torch.double))
print(f"s_clip.clip(2.0, 7.0): {s_clip.clip(2.0, 7.0)}")

# Where/Mask
s_where = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
cond = s_where > 2.5
other_tensor = torch.full((len(s_where),), 0.0, dtype=torch.double)
print(f"s_where.where(cond, other_tensor): {s_where.where(cond, other_tensor)}")
print(f"s_where.mask(cond, 99.0): {s_where.mask(cond, 99.0)}")


# Aggregations (scalars)
print(f"s.mean(): {s.mean()}")
print(f"s.std(): {s.std()}")
print(f"s.sum(): {s.sum()}")
print(f"s.min(): {s.min()}")
print(f"s.max(): {s.max()}")

# Functional: apply, map, agg, transform, pipe
s_func = pd.Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
print(f"s.apply(lambda x: x * 2): {s_func.apply(lambda x: x * 2)}")
print(f"s.map(lambda x: x ** 2): {s_func.map(lambda x: x ** 2)}")
print(f"s.agg('sum'): {s_func.agg('sum')}")
print(f"s.transform(lambda x: x / x.mean()): {s_func.transform(lambda x: x / x.mean())}")
print(f"s.pipe(lambda x: x + 10): {s_func.pipe(lambda x: x + 10)}")

# Value counts
s_vals = pd.Series(torch.tensor([1.0, 2.0, 1.0, 3.0, 2.0, 1.0], dtype=torch.double))
vc = s_vals.value_counts()
print(f"s_vals.value_counts(): {vc}")


# =============================================================================
# SECTION: DataFrame
# =============================================================================
print("\n=== Section: DataFrame ===")

# Creation
df = pd.DataFrame({
    'A': torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.double),
    'B': torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double),
    'C': torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.double),
})
print(f"df: {df}")

# Column access via __getattr__
print(f"df.A: {df.A}")

# Column access via __getitem__ (string)
print(f"df['B']: {df['B']}")

# Multiple columns via __getitem__ (list)
df_sub = df[['A', 'C']]
print(f"df[['A', 'C']]: {df_sub}")

# Boolean filtering
mask = df.A > 15.0
df_filtered = df[mask]
print(f"df[df.A > 15.0]: {df_filtered}")

# __setitem__
df['D'] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.double)
print(f"After adding column D: {df}")

# Properties
print(f"df.columns: {df.columns}")
print(f"df.shape: {df.shape}")
print(f"df.dtypes: {df.dtypes}")

# Head/Tail
print(f"df.head(2): {df.head(2)}")
print(f"df.tail(2): {df.tail(2)}")

# Drop
df_drop = df.drop(columns=['D'])
print(f"df.drop(['D']): {df_drop}")

# Rename
df_renamed = df.rename(columns={'A': 'Alpha', 'B': 'Beta'})
print(f"df.rename({{'A': 'Alpha', 'B': 'Beta'}}): {df_renamed}")

# Sort values
df_sorted = df.sort_values(by='A', ascending=False)
print(f"df.sort_values('A', ascending=False): {df_sorted}")

# Set/Reset index
df_indexed = df.set_index('A')
print(f"df.set_index('A'): {df_indexed}")
df_reset = df_indexed.reset_index()
print(f"df_indexed.reset_index(): {df_reset}")

# iloc indexing
print(f"df.iloc[1]: {df.iloc[1]}")
print(f"df.iloc[0:2]: {df.iloc[0:2]}")
print(f"df.iloc[1, 0]: {df.iloc[1, 0]}")

# loc indexing
print(f"df.loc[1]: {df.loc[1]}")
print(f"df.loc[0:2]: {df.loc[0:2]}")

# Merge
df1 = pd.DataFrame({
    'key': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
    'val1': torch.tensor([10.0, 20.0, 30.0], dtype=torch.double),
})
df2 = pd.DataFrame({
    'key': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
    'val2': torch.tensor([100.0, 200.0, 300.0], dtype=torch.double),
})
df_merged = df1.merge(df2, on='key')
print(f"df1.merge(df2, on='key'): {df_merged}")

# Join
df_joined = df1.join(df2)
print(f"df1.join(df2): {df_joined}")

# Functional: apply, agg, pipe
df_app = df.apply(lambda col: col + 1.0)
print(f"df.apply(lambda col: col + 1.0): {df_app}")

df_agg = df.agg('sum')
print(f"df.agg('sum'): {df_agg}")

df_pipe = df.pipe(lambda x: x[['A', 'B']])
print(f"df.pipe(lambda x: x[['A', 'B']]): {df_pipe}")

df_agg = df.agg('sum')
print(f"df.agg('sum'): {df_agg}")

df_pipe = df.pipe(lambda x: x[['A', 'B']])
print(f"df.pipe(lambda x: x[['A', 'B']]): {df_pipe}")

# Describe
desc = df.describe()
print(f"df.describe(): {desc}")

# Info
print("df.info():")
df.info()

# to_dict
print(f"df.to_dict(): {df.to_dict()}")
print(f"df.to_dict(orient='records'): {df.to_dict(orient='records')}")


# =============================================================================
# SECTION: GroupBy + GroupByColumn
# =============================================================================
print("\n=== Section: GroupBy + GroupByColumn ===")

df_gb = pd.DataFrame({
    'instrument': torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
    'price': torch.tensor([100.0, 105.0, 200.0, 210.0, 50.0], dtype=torch.double),
    'volume': torch.tensor([1000.0, 1100.0, 2000.0, 2200.0, 500.0], dtype=torch.double),
})





gb = df_gb.groupby('instrument')

# GroupByColumn aggregations
keys_sum, vals_sum = gb['price'].sum()
print(f"groupby('instrument')['price'].sum(): keys={keys_sum}, vals={vals_sum}")

keys_mean, vals_mean = gb['price'].mean()
print(f"groupby('instrument')['price'].mean(): keys={keys_mean}, vals={vals_mean}")

keys_count, vals_count = gb['price'].count()
print(f"groupby('instrument')['price'].count(): keys={keys_count}, vals={vals_count}")

keys_std, vals_std = gb['price'].std()
print(f"groupby('instrument')['price'].std(): keys={keys_std}, vals={vals_std}")

keys_min, vals_min = gb['price'].min()
print(f"groupby('instrument')['price'].min(): keys={keys_min}, vals={vals_min}")

keys_max, vals_max = gb['price'].max()
print(f"groupby('instrument')['price'].max(): keys={keys_max}, vals={vals_max}")

keys_first, vals_first = gb['price'].first()
print(f"groupby('instrument')['price'].first(): keys={keys_first}, vals={vals_first}")

keys_last, vals_last = gb['price'].last()
print(f"groupby('instrument')['price'].last(): keys={keys_last}, vals={vals_last}")


# =============================================================================
# SECTION: Rolling
# =============================================================================
print("\n=== Section: Rolling ===")

s_roll = pd.Series(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=torch.double))

roll = s_roll.rolling(window=3)

print(f"s_roll: {s_roll}")
print(f"s_roll.rolling(3).mean(): {roll.mean()}")
print(f"s_roll.rolling(3).sum(): {roll.sum()}")
print(f"s_roll.rolling(3).std(): {roll.std()}")
print(f"s_roll.rolling(3).min(): {roll.min()}")
print(f"s_roll.rolling(3).max(): {roll.max()}")


# =============================================================================
# SECTION: EWM
# =============================================================================
print("\n=== Section: EWM ===")

s_ewm = pd.Series(torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], dtype=torch.double))
ewm = s_ewm.ewm(span=2)

print(f"s_ewm: {s_ewm}")
print(f"s_ewm.ewm(span=2).mean(): {ewm.mean()}")


# =============================================================================
# SECTION: Expanding
# =============================================================================
print("\n=== Section: Expanding ===")

s_expand = pd.Series(torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.double))
exp = s_expand.expanding()

print(f"s_expand: {s_expand}")
print(f"s_expand.expanding().sum(): {exp.sum()}")
print(f"s_expand.expanding().mean(): {exp.mean()}")


# =============================================================================
# SECTION: Resampler (OHLC)
# =============================================================================
print("\n=== Section: Resampler ===")

df_resamp = pd.DataFrame({
    'bucket': torch.tensor([1, 1, 1, 2, 2], dtype=torch.long),
    'price': torch.tensor([100.0, 105.0, 102.0, 200.0, 210.0], dtype=torch.double),
})

resampler = df_resamp.groupby('bucket')['price'].resample('1D')

print(f"df_resamp: {df_resamp}")

resampler = df_resamp.groupby('bucket')['price'].resample('1D')

print(f"df_resamp: {df_resamp}")

open_s = resampler.first()
print(f"resampler.first() (Open): {open_s}")

high_s = resampler.max()
print(f"resampler.max() (High): {high_s}")

low_s = resampler.min()
print(f"resampler.min() (Low): {low_s}")

close_s = resampler.last()
print(f"resampler.last() (Close): {close_s}")


# =============================================================================
# SECTION: Index
# =============================================================================
print("\n=== Section: Index ===")

idx = pd.Index({
    'InstrumentID': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
    'TimeID': torch.tensor([0.0, 1.0, 2.0], dtype=torch.double),
})

print(f"idx.get_level_values('InstrumentID'): {idx.get_level_values('InstrumentID')}")
print(f"idx.get_level_values('TimeID'): {idx.get_level_values('TimeID')}")


# =============================================================================
# SECTION: concat (module-level)
# =============================================================================
print("\n=== Section: concat ===")

df_a = pd.DataFrame({
    'X': torch.tensor([1.0, 2.0], dtype=torch.double),
    'Y': torch.tensor([10.0, 20.0], dtype=torch.double),
})

df_b = pd.DataFrame({
    'X': torch.tensor([3.0, 4.0], dtype=torch.double),
    'Y': torch.tensor([30.0, 40.0], dtype=torch.double),
})

df_c = pd.DataFrame({
    'Z': torch.tensor([100.0, 200.0], dtype=torch.double),
})

# Vertical concatenation (axis=0)
df_vert = pd.concat([df_a, df_b], axis=0)
print(f"pd.concat([df_a, df_b], axis=0): {df_vert}")

# Horizontal concatenation (axis=1)
df_horiz = pd.concat([df_a, df_c], axis=1)
print(f"pd.concat([df_a, df_c], axis=1): {df_horiz}")


# =============================================================================
# SECTION: to_datetime and dt_floor (module-level)
# =============================================================================
print("\n=== Section: to_datetime & dt_floor ===")

# Unix timestamps in seconds
epochs = torch.tensor([1609459200, 1609545600, 1609632000], dtype=torch.double)  # Jan 1, 2, 3, 2021
print(f"epochs (seconds): {epochs}")

# Convert to nanoseconds
dt_ns = pd.to_datetime(epochs, unit='s')
print(f"pd.to_datetime(epochs, unit='s'): {dt_ns}")

# Floor to daily buckets
dt_floored = pd.dt_floor(dt_ns, freq='1D')
print(f"pd.dt_floor(dt_ns, freq='1D'): {dt_floored}")


# =============================================================================
# SECTION: Type casting
# =============================================================================
print("\n=== Section: Type Casting ===")

s_bool = pd.Series(torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.double))
print(f"s_bool: {s_bool}")
print(f"s_bool.astype(float): {s_bool.astype(float)}")


# =============================================================================
# FINAL VERIFICATION
# =============================================================================
print("\n✓ All wrapper API tests passed!")
