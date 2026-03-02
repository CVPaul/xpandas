"""pandas_migration.py — Side-by-side: pandas vs xpandas.

Shows common pandas patterns and their xpandas equivalents.
Run this script to verify all operations produce expected results.
"""

import xpandas as pd
import torch

# Set seed for reproducibility
torch.manual_seed(0)

print("=" * 70)
print("PANDAS vs XPANDAS MIGRATION GUIDE")
print("=" * 70)

# --- Section 1: DataFrame Creation ---
print("\n--- Section 1: DataFrame Creation ---")
print("# pandas: df = pandas.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})")
df = pd.DataFrame({
    'a': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
    'b': torch.tensor([4.0, 5.0, 6.0], dtype=torch.double)
})
print("xpandas result:")
print(df)
print()

# --- Section 2: Column Access (df['col'] and df.col) ---
print("--- Section 2: Column Access ---")
print("# pandas: col_a = df['a']")
col_a = df['a']
print("xpandas df['a']:")
print(col_a)
print("# pandas: col_b = df.b")
col_b = df.b
print("xpandas df.b:")
print(col_b)
print()

# --- Section 3: Arithmetic Operations ---
print("--- Section 3: Arithmetic Operations ---")
print("# pandas: result = df['a'] + df['b']")
result_add = df['a'] + df['b']
print("xpandas df['a'] + df['b']:")
print(result_add)
print("# pandas: result = df['a'] * 2")
result_mul = df['a'] * 2.0
print("xpandas df['a'] * 2.0:")
print(result_mul)
print("# pandas: result = df['a'] / df['b']")
result_div = df['a'] / df['b']
print("xpandas df['a'] / df['b']:")
print(result_div)
print()

# --- Section 4: Comparison Operations ---
print("--- Section 4: Comparison Operations ---")
print("# pandas: mask = df['a'] > 1.5")
mask = df['a'] > 1.5
print("xpandas df['a'] > 1.5:")
print(mask)
print("# pandas: mask = df['b'] >= df['a']")
mask2 = df['b'] >= df['a']
print("xpandas df['b'] >= df['a']:")
print(mask2)
print()

# --- Section 5: GroupBy Aggregation ---
print("--- Section 5: GroupBy Aggregation ---")
# Create a DataFrame with group keys for groupby
df_grouped = pd.DataFrame({
    'group': torch.tensor([1, 1, 2, 2, 3], dtype=torch.long),
    'value': torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.double)
})
print("DataFrame for groupby:")
print(df_grouped)
print("# pandas: result = df.groupby('group')['value'].sum()")
print("# xpandas returns (keys_tensor, values_tensor) tuple")
keys, sums = df_grouped.groupby('group')['value'].sum()
print("xpandas groupby('group')['value'].sum():")
print(f"  Keys: {keys}")
print(f"  Sums: {sums}")
print()

# --- Section 6: Rolling Window ---
print("--- Section 6: Rolling Window Mean ---")
series = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
print("Original series:")
print(series)
print("# pandas: result = series.rolling(3).mean()")
rolling_mean = series.rolling(3).mean()
print("xpandas series.rolling(3).mean():")
print(rolling_mean)
print()

# --- Section 7: Exponential Weighted Mean ---
print("--- Section 7: Exponential Weighted Mean (EWM) ---")
print("# pandas: result = series.ewm(span=3).mean()")
ewm_result = series.ewm(span=3).mean()
print("xpandas series.ewm(span=3).mean():")
print(ewm_result)
print()

# --- Section 8: fillna, shift, and pct_change ---
print("--- Section 8: Data Cleaning (fillna, shift, pct_change) ---")
series_with_nan = pd.Series(torch.tensor([1.0, float('nan'), 3.0, 4.0, 5.0], dtype=torch.double))
print("Series with NaN:")
print(series_with_nan)
print("# pandas: result = series.fillna(0.0)")
filled = series_with_nan.fillna(0.0)
print("xpandas fillna(0.0):")
print(filled)
print("# pandas: result = series.shift(1)")
shifted = series.shift(1)
print("xpandas series.shift(1):")
print(shifted)
print("# pandas: result = series.pct_change()")
pct = series.pct_change()
print("xpandas series.pct_change():")
print(pct)
print()

# --- Section 9: sort_values ---
print("--- Section 9: Sort Values (DataFrame) ---")
unsorted_df = pd.DataFrame({
    'name': torch.tensor([5.0, 2.0, 8.0, 1.0, 9.0], dtype=torch.double)
})
print("Unsorted DataFrame:")
print(unsorted_df)
print("# pandas: result = df.sort_values(by='name')")
sorted_asc = unsorted_df.sort_values(by='name')
print("xpandas sort_values(by='name'):")
print(sorted_asc)
print("# pandas: result = df.sort_values(by='name', ascending=False)")
sorted_desc = unsorted_df.sort_values(by='name', ascending=False)
print("xpandas sort_values(by='name', ascending=False):")
print(sorted_desc)
print()

# --- Section 10: merge (Inner Join) ---
print("--- Section 10: Merge (Inner Join) ---")
df1 = pd.DataFrame({
    'key': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double),
    'val1': torch.tensor([10.0, 20.0, 30.0], dtype=torch.double)
})
df2 = pd.DataFrame({
    'key': torch.tensor([2.0, 3.0, 4.0], dtype=torch.double),
    'val2': torch.tensor([200.0, 300.0, 400.0], dtype=torch.double)
})
print("DataFrame 1:")
print(df1)
print("DataFrame 2:")
print(df2)
print("# pandas: result = pd.merge(df1, df2, on='key', how='inner')")
merged = df1.merge(df2, on='key', how='inner')
print("xpandas merge result:")
print(merged)
print()

# --- Section 11: Type Casting (astype) ---
print("--- Section 11: Type Casting (astype) ---")
series_int_like = pd.Series(torch.tensor([1.0, 2.0, 3.0], dtype=torch.double))
print("Original series (dtype=torch.double):")
print(series_int_like)
print("# pandas: result = series.astype(float)")
print("# xpandas: astype preserves torch.double (xpandas native dtype)")
casted = series_int_like.astype(float)
print("xpandas astype(float):")
print(casted)
print()

# --- Section 12: where / mask ---
print("--- Section 12: where / mask Operations ---")
print("# pandas: result = series.where(series > 2.0, -1.0)")
series_for_mask = pd.Series(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.double))
print("Original series:")
print(series_for_mask)
where_result = series_for_mask.where(series_for_mask > 2.0, torch.full((5,), -1.0, dtype=torch.double))
print("xpandas where(series > 2.0, -1.0):")
print(where_result)
print()

# --- Summary ---
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
KEY DIFFERENCES:
1. Import: Use 'import xpandas as pd' instead of pandas
2. Tensors: Always use torch.tensor() instead of Python lists
3. dtype: xpandas works with torch.double (64-bit float) by default
4. GroupBy: Returns (keys_tensor, values_tensor) tuple, NOT a pandas DataFrame
5. NaN: Use float('nan') for missing values in torch tensors
6. API: Most pandas methods have direct xpandas equivalents
   - Series: df['col'] or df.col → returns Series
   - Arithmetic: +, -, *, / work as expected
   - Aggregations: .sum(), .mean(), .min(), .max()
   - Window: .rolling(), .ewm(), .shift()
   - Transforms: .fillna(), .pct_change(), .sort_values()
   - Joins: .merge() with on= and how= parameters

MIGRATION PATTERN:
  Before: df = pandas.DataFrame({'a': [1, 2, 3]})
  After:  df = pd.DataFrame({'a': torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)})
""")
print("=" * 70)
