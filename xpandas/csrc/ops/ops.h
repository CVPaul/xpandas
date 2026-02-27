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

} // namespace xpandas
