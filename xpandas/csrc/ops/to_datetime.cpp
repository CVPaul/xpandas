/**
 * to_datetime.cpp -- epoch timestamp conversion and datetime floor.
 *
 * Implements two ops:
 *
 *   to_datetime(Tensor epochs, str unit) -> Tensor
 *       Convert epoch timestamps to nanosecond epoch (int64),
 *       matching pandas datetime64[ns] internal representation.
 *       Supported units: "s" (seconds), "ms" (milliseconds),
 *       "us" (microseconds), "ns" (nanoseconds).
 *
 *   dt_floor(Tensor dt_ns, int interval_ns) -> Tensor
 *       Floor datetime (nanosecond epoch) to the given interval.
 *       Common intervals: 86400000000000 (1D), 3600000000000 (1h),
 *       60000000000 (1min), 1000000000 (1s).
 *       Result can be used directly as a groupby key.
 */

#include "ops.h"

#include <cmath>
#include <string>

namespace xpandas {

at::Tensor to_datetime(const at::Tensor& epochs, const std::string& unit) {
    TORCH_CHECK(epochs.dim() == 1,
                "to_datetime: input must be 1-D, got ", epochs.dim(), "-D");

    // Determine the multiplier to convert to nanoseconds
    int64_t multiplier = 0;
    if (unit == "s") {
        multiplier = 1000000000LL;       // 1e9
    } else if (unit == "ms") {
        multiplier = 1000000LL;          // 1e6
    } else if (unit == "us") {
        multiplier = 1000LL;             // 1e3
    } else if (unit == "ns") {
        multiplier = 1LL;
    } else {
        TORCH_CHECK(false, "to_datetime: unsupported unit '", unit,
                     "', expected one of: s, ms, us, ns");
    }

    // If input is float64, convert to int64 first (truncate toward zero)
    at::Tensor input_i64;
    if (epochs.scalar_type() == at::kDouble || epochs.scalar_type() == at::kFloat) {
        input_i64 = epochs.to(at::kLong);
    } else {
        TORCH_CHECK(epochs.scalar_type() == at::kLong,
                    "to_datetime: input must be int64 or float64, got ",
                    epochs.scalar_type());
        input_i64 = epochs;
    }

    if (multiplier == 1) {
        return input_i64.clone();
    }

    // Multiply to nanoseconds
    return input_i64 * multiplier;
}


at::Tensor dt_floor(const at::Tensor& dt_ns, int64_t interval_ns) {
    TORCH_CHECK(dt_ns.dim() == 1,
                "dt_floor: input must be 1-D, got ", dt_ns.dim(), "-D");
    TORCH_CHECK(dt_ns.scalar_type() == at::kLong,
                "dt_floor: input must be int64 (nanosecond epoch)");
    TORCH_CHECK(interval_ns > 0,
                "dt_floor: interval_ns must be positive, got ", interval_ns);

    const int64_t n = dt_ns.size(0);
    if (n == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kLong));
    }

    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kLong));
    auto in_a  = dt_ns.accessor<int64_t, 1>();
    auto out_a = result.accessor<int64_t, 1>();

    for (int64_t i = 0; i < n; ++i) {
        int64_t v = in_a[i];
        // Floor division that handles negative timestamps correctly
        // (timestamps before 1970-01-01)
        if (v >= 0) {
            out_a[i] = (v / interval_ns) * interval_ns;
        } else {
            out_a[i] = ((v - interval_ns + 1) / interval_ns) * interval_ns;
        }
    }

    return result;
}

} // namespace xpandas
