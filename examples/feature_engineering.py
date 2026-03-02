#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_engineering.py — Generic feature engineering pipeline using xpandas.

Demonstrates non-financial feature engineering techniques using xpandas wrapper API.
Examples include sensor data normalization, website metrics aggregation, and general-purpose
statistical transformations like rolling stats, z-score normalization, and cumulative operations.

Operations shown:
    - rolling().mean(), .std(), .min(), .max()
    - ewm().mean() (exponential weighted moving average)
    - expanding().sum(), .mean()
    - zscore() (standardization)
    - rank() (ranking)
    - fillna() (missing value imputation)
    - shift() (lagging)
    - pct_change() (percentage change)
    - cumsum(), cumprod() (cumulative operations)
    - clip() (value bounding)
    - abs() (absolute value)
    - log() (logarithmic transformation)

Data model (synthetic sensor & web metrics):
    - "timestamp": epoch timestamp (int64)
    - "sensor_temperature": float64 values (°C)
    - "sensor_humidity": float64 values (%)
    - "website_requests": float64 values (requests/sec)
"""

import torch
import xpandas as pd

torch.manual_seed(42)

# ============================================================================
# 1. Generate synthetic sensor and web metrics data
# ============================================================================

print("=" * 70)
print("XPANDAS FEATURE ENGINEERING DEMO")
print("=" * 70)

# Simulate 100 time steps of sensor data
n = 100
device = torch.device("cpu")

# Create base synthetic data
timestamps = torch.arange(1000000, 1000000 + n, dtype=torch.float64, device=device)
temperature_base = 20.0 + 5.0 * torch.sin(torch.arange(n, dtype=torch.float64) * 0.1)
temperature_noise = torch.randn(n, dtype=torch.float64, device=device) * 0.5
sensor_temperature = temperature_base + temperature_noise

humidity_base = 50.0 + 30.0 * torch.cos(torch.arange(n, dtype=torch.float64) * 0.15)
humidity_noise = torch.randn(n, dtype=torch.float64, device=device) * 2.0
sensor_humidity = humidity_base + humidity_noise

website_requests = 1000.0 + 200.0 * torch.sin(torch.arange(n, dtype=torch.float64) * 0.2)
website_noise = torch.randn(n, dtype=torch.float64, device=device) * 50.0
website_requests = website_requests + website_noise

# Create DataFrame
df = pd.DataFrame({
    "timestamp": timestamps,
    "sensor_temperature": sensor_temperature,
    "sensor_humidity": sensor_humidity,
    "website_requests": website_requests,
})

print("\n[1] Original DataFrame (first 10 rows):")
print(f"    Shape: {df.shape}")
print(f"    Columns: {list(df._data.keys())}")
print(f"    Temperature [0:5]: {sensor_temperature[:5]}")
print(f"    Humidity [0:5]: {sensor_humidity[:5]}")
print(f"    Requests [0:5]: {website_requests[:5]}")

# ============================================================================
# 2. Z-score normalization (standardization)
# ============================================================================
print("\n[2] Z-score normalization (standardization):")
temp_zscore = df.sensor_temperature.zscore()
print(f"    Temperature zscore [0:5]: {temp_zscore.values[:5]}")
print(f"    Mean of zscore: {temp_zscore.mean():.6f} (should be ~0)")
print(f"    Std of zscore: {temp_zscore.std():.6f} (should be ~1)")

# ============================================================================
# 3. Rolling statistics (5-step window)
# ============================================================================
print("\n[3] Rolling statistics (window=5):")
temp_rolling_mean = df.sensor_temperature.rolling(5).mean()
temp_rolling_std = df.sensor_temperature.rolling(5).std()
temp_rolling_min = df.sensor_temperature.rolling(5).min()
temp_rolling_max = df.sensor_temperature.rolling(5).max()
print(f"    Rolling mean [5:10]: {temp_rolling_mean.values[5:10]}")
print(f"    Rolling std [5:10]: {temp_rolling_std.values[5:10]}")
print(f"    Rolling min [5:10]: {temp_rolling_min.values[5:10]}")
print(f"    Rolling max [5:10]: {temp_rolling_max.values[5:10]}")

# ============================================================================
# 4. Exponential Weighted Moving Average (EWM)
# ============================================================================
print("\n[4] Exponential Weighted Moving Average (span=10):")
temp_ewm = df.sensor_temperature.ewm(span=10).mean()
print(f"    EWM mean [0:5]: {temp_ewm.values[0:5]}")
print(f"    EWM mean [95:100]: {temp_ewm.values[95:100]}")

# ============================================================================
# 5. Expanding operations
# ============================================================================
print("\n[5] Expanding operations (cumulative):")
requests_expanding_sum = df.website_requests.expanding().sum()
requests_expanding_mean = df.website_requests.expanding().mean()
print(f"    Expanding sum [0:5]: {requests_expanding_sum.values[0:5]}")
print(f"    Expanding mean [0:5]: {requests_expanding_mean.values[0:5]}")
print(f"    Expanding sum [98:100]: {requests_expanding_sum.values[98:100]}")

# ============================================================================
# 6. Ranking (percentile ranking)
# ============================================================================
print("\n[6] Ranking (percentile ranking):")
humidity_rank = df.sensor_humidity.rank()
print(f"    Humidity ranks [0:10]: {humidity_rank.values[0:10]}")
print(f"    Min rank: {humidity_rank.min():.1f}, Max rank: {humidity_rank.max():.1f}")

# ============================================================================
# 7. Fillna (forward fill with constant)
# ============================================================================
print("\n[7] Fillna (impute missing with constant):")
# Create tensor with some NaNs (simulate missing data)
temp_with_nans_data = df.sensor_temperature.values.clone()
temp_with_nans_data[10] = float('nan')
temp_with_nans_data[25] = float('nan')
temp_with_nans_data[50] = float('nan')
temp_with_nans = pd.Series(temp_with_nans_data)

temp_filled = temp_with_nans.fillna(20.0)
print(f"    Original with NaNs [8:12]: {temp_with_nans.values[8:12]}")
print(f"    After fillna(20.0) [8:12]: {temp_filled.values[8:12]}")

# ============================================================================
# 8. Shift (lag/lead operations)
# ============================================================================
print("\n[8] Shift (lag by 3 steps):")
temp_lagged = df.sensor_temperature.shift(periods=3)
print(f"    Original [3:8]: {df.sensor_temperature.values[3:8]}")
print(f"    Lagged by 3 [3:8]: {temp_lagged.values[3:8]}")
print(f"    Note: First 3 values are NaN after shift")

# ============================================================================
# 9. Percentage change (rate of change)
# ============================================================================
print("\n[9] Percentage change (1-step pct_change):")
requests_pct_change = df.website_requests.pct_change(periods=1)
print(f"    Website requests [0:5]: {df.website_requests.values[0:5]}")
print(f"    Pct change [1:5]: {requests_pct_change.values[1:5]}")
print(f"    Note: First value is NaN")

# ============================================================================
# 10. Cumulative sum and cumulative product
# ============================================================================
print("\n[10] Cumulative operations:")
requests_cumsum = df.website_requests.cumsum()
# Normalize for cumprod to be reasonable (use small positive values)
requests_normalized = (df.website_requests - df.website_requests.min()) / (df.website_requests.max() - df.website_requests.min())
requests_normalized = requests_normalized + 0.5  # Shift to [0.5, 1.5] for cumprod
requests_cumprod = requests_normalized.cumprod()
print(f"    Website requests cumsum [0:5]: {requests_cumsum.values[0:5]}")
print(f"    Website requests cumsum [95:100]: {requests_cumsum.values[95:100]}")
print(f"    Normalized cumprod [0:5]: {requests_cumprod.values[0:5]}")
print(f"    Normalized cumprod [95:100]: {requests_cumprod.values[95:100]}")

# ============================================================================
# 11. Clip (bounding values)
# ============================================================================
print("\n[11] Clip (bound values to [18, 22]):")
temp_clipped = df.sensor_temperature.clip(lower=18.0, upper=22.0)
print(f"    Original temp [0:10]: {df.sensor_temperature.values[0:10]}")
print(f"    Clipped to [18, 22] [0:10]: {temp_clipped.values[0:10]}")

# ============================================================================
# 12. Absolute value
# ============================================================================
print("\n[12] Absolute value:")
temp_centered = df.sensor_temperature - df.sensor_temperature.mean()
temp_abs = temp_centered.abs()
print(f"    Centered temp [0:5]: {temp_centered.values[0:5]}")
print(f"    Absolute value [0:5]: {temp_abs.values[0:5]}")

# ============================================================================
# 13. Logarithmic transformation
# ============================================================================
print("\n[13] Logarithmic transformation (log of positive values):")
requests_positive = df.website_requests + 1.0  # Ensure positive
requests_log = requests_positive.log()
print(f"    Website requests [0:5]: {df.website_requests.values[0:5]}")
print(f"    Log(requests+1) [0:5]: {requests_log.values[0:5]}")

# ============================================================================
# 14. Composite feature: Rolling mean reversion signal
# ============================================================================
print("\n[14] Composite feature: Mean reversion signal")
print("    Calculate: zscore of (value - rolling_mean) / rolling_std")
temp_mean = df.sensor_temperature.rolling(window=10).mean()
temp_std = df.sensor_temperature.rolling(window=10).std()
temp_deviation = (df.sensor_temperature - temp_mean)
# Avoid division by very small std; clip std to minimum
temp_std_safe = temp_std.clip(lower=0.1, upper=float('inf'))
temp_deviation_normalized = temp_deviation / temp_std_safe
print(f"    Temperature deviation (normalized) [10:20]: {temp_deviation_normalized.values[10:20]}")

# ============================================================================
# 15. Summary statistics
# ============================================================================
print("\n[15] Summary statistics:")
print(f"    DataFrame shape: {df.shape}")
print(f"    DataFrame columns: {list(df._data.keys())}")
print(f"    Temperature: mean={df.sensor_temperature.mean():.2f}, std={df.sensor_temperature.std():.2f}")
print(f"    Humidity: mean={df.sensor_humidity.mean():.2f}, std={df.sensor_humidity.std():.2f}")
print(f"    Requests: mean={df.website_requests.mean():.2f}, std={df.website_requests.std():.2f}")

print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print("Features demonstrated:")
print("  ✓ rolling().mean(), .std(), .min(), .max()")
print("  ✓ ewm().mean()")
print("  ✓ expanding().sum(), .mean()")
print("  ✓ zscore()")
print("  ✓ rank()")
print("  ✓ fillna()")
print("  ✓ shift()")
print("  ✓ pct_change()")
print("  ✓ cumsum(), cumprod()")
print("  ✓ clip()")
print("  ✓ abs()")
print("  ✓ log()")
print("=" * 70)
