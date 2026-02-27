/**
 * rolling.cpp -- Rolling window operations.
 *
 * Ops:
 *   rolling_mean(x, window) -> Tensor   -- rolling mean with min_periods=window
 *   rolling_sum(x, window)  -> Tensor   -- rolling sum with min_periods=window
 *   rolling_std(x, window)  -> Tensor   -- rolling std (ddof=1) with min_periods=window
 *
 * All ops produce NaN for positions where the window is not yet full
 * (i.e., the first `window-1` elements), matching pandas default
 * min_periods=window behavior.
 */

#include "ops.h"

#include <cmath>
#include <limits>

namespace xpandas {

at::Tensor rolling_sum(const at::Tensor& x, int64_t window) {
    TORCH_CHECK(x.dim() == 1, "rolling_sum: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rolling_sum: input must be float64");
    TORCH_CHECK(window > 0, "rolling_sum: window must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    auto x_a = x.accessor<double, 1>();
    auto r_a = result.accessor<double, 1>();

    // Fill first window-1 positions with NaN
    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        r_a[i] = std::numeric_limits<double>::quiet_NaN();
    }

    if (n < window) return result;

    // Compute initial window sum
    double sum = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        sum += x_a[i];
    }
    r_a[window - 1] = sum;

    // Slide the window
    for (int64_t i = window; i < n; ++i) {
        sum += x_a[i] - x_a[i - window];
        r_a[i] = sum;
    }

    return result;
}

at::Tensor rolling_mean(const at::Tensor& x, int64_t window) {
    TORCH_CHECK(x.dim() == 1, "rolling_mean: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rolling_mean: input must be float64");
    TORCH_CHECK(window > 0, "rolling_mean: window must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    auto x_a = x.accessor<double, 1>();
    auto r_a = result.accessor<double, 1>();

    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        r_a[i] = std::numeric_limits<double>::quiet_NaN();
    }

    if (n < window) return result;

    double sum = 0.0;
    double dw = static_cast<double>(window);
    for (int64_t i = 0; i < window; ++i) {
        sum += x_a[i];
    }
    r_a[window - 1] = sum / dw;

    for (int64_t i = window; i < n; ++i) {
        sum += x_a[i] - x_a[i - window];
        r_a[i] = sum / dw;
    }

    return result;
}

at::Tensor rolling_std(const at::Tensor& x, int64_t window) {
    TORCH_CHECK(x.dim() == 1, "rolling_std: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rolling_std: input must be float64");
    TORCH_CHECK(window > 0, "rolling_std: window must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    auto x_a = x.accessor<double, 1>();
    auto r_a = result.accessor<double, 1>();

    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        r_a[i] = std::numeric_limits<double>::quiet_NaN();
    }

    if (n < window) return result;

    // Welford-style online computation for numerical stability
    // For the initial window, compute mean and M2
    double sum = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        sum += x_a[i];
    }
    double mean = sum / static_cast<double>(window);

    double m2 = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        double d = x_a[i] - mean;
        m2 += d * d;
    }

    double dw = static_cast<double>(window);
    if (window <= 1) {
        r_a[window - 1] = std::numeric_limits<double>::quiet_NaN();
    } else {
        r_a[window - 1] = std::sqrt(m2 / (dw - 1.0));
    }

    // Slide the window: update sum, mean, and recompute variance
    // (Using the two-pass approach for each slide to ensure numerical accuracy
    //  is sufficient for typical quant data sizes. For very large windows,
    //  an online algorithm could be used.)
    for (int64_t i = window; i < n; ++i) {
        sum += x_a[i] - x_a[i - window];
        mean = sum / dw;

        m2 = 0.0;
        for (int64_t j = i - window + 1; j <= i; ++j) {
            double d = x_a[j] - mean;
            m2 += d * d;
        }
        r_a[i] = std::sqrt(m2 / (dw - 1.0));
    }

    return result;
}

} // namespace xpandas
