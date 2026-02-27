/**
 * xpandas/csrc/ops/ops.h -- Common header for xpandas custom ops.
 *
 * Each op is declared here so that register.cpp can reference them
 * without including every individual .cpp file.
 */

#pragma once

#include <torch/library.h>
#include <ATen/ATen.h>

#include <string>
#include <tuple>

namespace xpandas {

// groupby_resample_ohlc.cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
groupby_resample_ohlc(const at::Tensor& key, const at::Tensor& value);

// compare.cpp
at::Tensor compare_gt(const at::Tensor& a, const at::Tensor& b);
at::Tensor compare_lt(const at::Tensor& a, const at::Tensor& b);

// cast.cpp
at::Tensor bool_to_float(const at::Tensor& x);

// lookup.cpp
at::Tensor lookup(
    const c10::Dict<std::string, at::Tensor>& table,
    const std::string& key);

// breakout_signal.cpp
at::Tensor breakout_signal(
    const at::Tensor& price,
    const at::Tensor& high,
    const at::Tensor& low);

// rank.cpp
at::Tensor rank(const at::Tensor& x);

// to_datetime.cpp
at::Tensor to_datetime(const at::Tensor& epochs, const std::string& unit);
at::Tensor dt_floor(const at::Tensor& dt_ns, int64_t interval_ns);

// groupby_agg.cpp
std::tuple<at::Tensor, at::Tensor>
groupby_sum(const at::Tensor& key, const at::Tensor& value);
std::tuple<at::Tensor, at::Tensor>
groupby_mean(const at::Tensor& key, const at::Tensor& value);
std::tuple<at::Tensor, at::Tensor>
groupby_count(const at::Tensor& key, const at::Tensor& value);
std::tuple<at::Tensor, at::Tensor>
groupby_std(const at::Tensor& key, const at::Tensor& value);

// rolling.cpp
at::Tensor rolling_sum(const at::Tensor& x, int64_t window);
at::Tensor rolling_mean(const at::Tensor& x, int64_t window);
at::Tensor rolling_std(const at::Tensor& x, int64_t window);

// shift.cpp
at::Tensor shift(const at::Tensor& x, int64_t periods);

// fillna.cpp
at::Tensor fillna(const at::Tensor& x, double fill_value);

// where.cpp
at::Tensor where_(const at::Tensor& cond, const at::Tensor& x,
                  const at::Tensor& other);
at::Tensor masked_fill(const at::Tensor& x, const at::Tensor& mask,
                       double fill_value);

// pct_change.cpp
at::Tensor pct_change(const at::Tensor& x, int64_t periods);

// cumulative.cpp
at::Tensor cumsum(const at::Tensor& x);
at::Tensor cumprod(const at::Tensor& x);

// clip.cpp
at::Tensor clip(const at::Tensor& x, double lower, double upper);

} // namespace xpandas
