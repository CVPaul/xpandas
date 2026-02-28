/**
 * rolling_minmax.cpp -- Rolling window min and max operations.
 *
 * Ops:
 *   rolling_min(x, window) -> Tensor  -- rolling minimum, min_periods=window
 *   rolling_max(x, window) -> Tensor  -- rolling maximum, min_periods=window
 *
 * Uses a monotonic deque approach for O(n) complexity instead of O(n*w).
 * First window-1 elements are NaN (matching pandas min_periods=window).
 */

#include "ops.h"

#include <cmath>
#include <deque>
#include <limits>

namespace xpandas {

at::Tensor rolling_min(const at::Tensor& x, int64_t window) {
    TORCH_CHECK(x.dim() == 1, "rolling_min: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rolling_min: input must be float64");
    TORCH_CHECK(window > 0, "rolling_min: window must be positive");

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

    // Monotonic deque: stores indices; front is always the min in the window
    std::deque<int64_t> dq;

    for (int64_t i = 0; i < n; ++i) {
        // Remove elements that are out of the window
        while (!dq.empty() && dq.front() <= i - window) {
            dq.pop_front();
        }
        // Remove elements from the back that are >= current (maintain ascending)
        while (!dq.empty() && px[dq.back()] >= px[i]) {
            dq.pop_back();
        }
        dq.push_back(i);

        if (i >= window - 1) {
            pr[i] = px[dq.front()];
        }
    }

    return result;
}

at::Tensor rolling_max(const at::Tensor& x, int64_t window) {
    TORCH_CHECK(x.dim() == 1, "rolling_max: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rolling_max: input must be float64");
    TORCH_CHECK(window > 0, "rolling_max: window must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

    for (int64_t i = 0; i < std::min(window - 1, n); ++i) {
        pr[i] = NaN;
    }

    if (n < window) return result;

    // Monotonic deque: stores indices; front is always the max in the window
    std::deque<int64_t> dq;

    for (int64_t i = 0; i < n; ++i) {
        while (!dq.empty() && dq.front() <= i - window) {
            dq.pop_front();
        }
        // Remove elements from the back that are <= current (maintain descending)
        while (!dq.empty() && px[dq.back()] <= px[i]) {
            dq.pop_back();
        }
        dq.push_back(i);

        if (i >= window - 1) {
            pr[i] = px[dq.front()];
        }
    }

    return result;
}

} // namespace xpandas
