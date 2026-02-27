/**
 * shift.cpp -- Shift (lag/lead) operation.
 *
 * Equivalent to pandas Series.shift(periods).
 *
 * Positive periods shift forward (introduce NaN at the beginning).
 * Negative periods shift backward (introduce NaN at the end).
 */

#include "ops.h"

#include <cmath>
#include <limits>

namespace xpandas {

at::Tensor shift(const at::Tensor& x, int64_t periods) {
    TORCH_CHECK(x.dim() == 1, "shift: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "shift: input must be float64");

    const int64_t n = x.size(0);
    at::Tensor result = at::full({n}, std::numeric_limits<double>::quiet_NaN(),
                                 at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    auto x_a = x.accessor<double, 1>();
    auto r_a = result.accessor<double, 1>();

    if (periods >= 0) {
        // Shift forward: result[i] = x[i - periods] for i >= periods
        int64_t p = std::min(periods, n);
        for (int64_t i = p; i < n; ++i) {
            r_a[i] = x_a[i - p];
        }
    } else {
        // Shift backward: result[i] = x[i - periods] for i < n + periods
        int64_t p = std::min(-periods, n);
        for (int64_t i = 0; i < n - p; ++i) {
            r_a[i] = x_a[i + p];
        }
    }

    return result;
}

} // namespace xpandas
