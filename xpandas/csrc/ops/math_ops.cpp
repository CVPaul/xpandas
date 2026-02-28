/**
 * math_ops.cpp -- Element-wise math operations.
 *
 * Ops:
 *   abs_(x)      -> Tensor  -- element-wise absolute value
 *   log_(x)      -> Tensor  -- element-wise natural log (NaN for x <= 0)
 *   zscore(x)    -> Tensor  -- (x - mean) / std (population zscore)
 *
 * All ops use raw data pointers for minimal overhead.
 */

#include "ops.h"

#include <cmath>
#include <limits>

namespace xpandas {

at::Tensor abs_(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "abs_: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "abs_: input must be float64");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    for (int64_t i = 0; i < n; ++i) {
        pr[i] = std::abs(px[i]);
    }

    return result;
}

at::Tensor log_(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "log_: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "log_: input must be float64");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    for (int64_t i = 0; i < n; ++i) {
        pr[i] = (px[i] > 0.0) ? std::log(px[i]) : NaN;
    }

    return result;
}

at::Tensor zscore(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "zscore: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "zscore: input must be float64");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // Two-pass: compute mean, then std, then normalize.
    // For small quant data this is fine.
    // Skip NaN values in computation.
    double sum = 0.0;
    int64_t count = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isnan(px[i])) {
            sum += px[i];
            count += 1;
        }
    }

    if (count <= 1) {
        // Can't compute std with 0 or 1 non-NaN elements
        for (int64_t i = 0; i < n; ++i) {
            pr[i] = NaN;
        }
        return result;
    }

    double mean = sum / static_cast<double>(count);

    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        if (!std::isnan(px[i])) {
            double d = px[i] - mean;
            sum_sq += d * d;
        }
    }

    // ddof=1 to match pandas default
    double std_val = std::sqrt(sum_sq / static_cast<double>(count - 1));

    if (std_val < 1e-15) {
        // All values are the same — zscore is 0
        for (int64_t i = 0; i < n; ++i) {
            pr[i] = std::isnan(px[i]) ? NaN : 0.0;
        }
        return result;
    }

    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(px[i])) {
            pr[i] = NaN;
        } else {
            pr[i] = (px[i] - mean) / std_val;
        }
    }

    return result;
}

} // namespace xpandas
