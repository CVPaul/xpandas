/**
 * pct_change.cpp -- Percentage change.
 *
 * Equivalent to pandas Series.pct_change(periods=1).
 * Computes (x[i] - x[i-periods]) / x[i-periods] for each element.
 * The first `periods` elements are NaN.
 */

#include "ops.h"

#include <cmath>
#include <limits>

namespace xpandas {

at::Tensor pct_change(const at::Tensor& x, int64_t periods) {
    TORCH_CHECK(x.dim() == 1, "pct_change: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "pct_change: input must be float64");
    TORCH_CHECK(periods > 0, "pct_change: periods must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::full({n}, std::numeric_limits<double>::quiet_NaN(),
                                 at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    auto x_a = x.accessor<double, 1>();
    auto r_a = result.accessor<double, 1>();

    for (int64_t i = periods; i < n; ++i) {
        double prev = x_a[i - periods];
        if (prev == 0.0) {
            r_a[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            r_a[i] = (x_a[i] - prev) / prev;
        }
    }

    return result;
}

} // namespace xpandas
