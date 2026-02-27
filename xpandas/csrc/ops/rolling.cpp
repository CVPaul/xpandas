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
 *
 * Optimized: uses raw data pointers instead of accessors; rolling_std uses
 * an O(n) online algorithm instead of O(n*w) recomputation.
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

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    // Fill first window-1 positions with NaN
    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        pr[i] = NaN;
    }

    if (n < window) return result;

    // Compute initial window sum
    double sum = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        sum += px[i];
    }
    pr[window - 1] = sum;

    // Slide the window
    for (int64_t i = window; i < n; ++i) {
        sum += px[i] - px[i - window];
        pr[i] = sum;
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

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    const double dw = static_cast<double>(window);

    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        pr[i] = NaN;
    }

    if (n < window) return result;

    double sum = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        sum += px[i];
    }
    pr[window - 1] = sum / dw;

    for (int64_t i = window; i < n; ++i) {
        sum += px[i] - px[i - window];
        pr[i] = sum / dw;
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

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    const double dw = static_cast<double>(window);

    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        pr[i] = NaN;
    }

    if (n < window) return result;

    // O(n) online rolling variance using sum and sum-of-squares.
    // Variance = (sum_sq - sum*sum/w) / (w-1)  [Bessel's correction, ddof=1]
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < window; ++i) {
        sum    += px[i];
        sum_sq += px[i] * px[i];
    }

    if (window <= 1) {
        pr[window - 1] = NaN;
    } else {
        double var = (sum_sq - sum * sum / dw) / (dw - 1.0);
        pr[window - 1] = std::sqrt(std::max(var, 0.0));
    }

    // Slide the window
    for (int64_t i = window; i < n; ++i) {
        double x_old = px[i - window];
        double x_new = px[i];
        sum    += x_new - x_old;
        sum_sq += x_new * x_new - x_old * x_old;

        double var = (sum_sq - sum * sum / dw) / (dw - 1.0);
        // Guard against numerical noise producing tiny negative variance
        pr[i] = std::sqrt(std::max(var, 0.0));
    }

    return result;
}

} // namespace xpandas
