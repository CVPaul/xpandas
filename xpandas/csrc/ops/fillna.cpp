/**
 * fillna.cpp -- Fill NaN values.
 *
 * Equivalent to pandas Series.fillna(value).
 * Replaces all NaN entries with the given fill_value.
 */

#include "ops.h"

#include <cmath>

namespace xpandas {

at::Tensor fillna(const at::Tensor& x, double fill_value) {
    TORCH_CHECK(x.dim() == 1, "fillna: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "fillna: input must be float64");

    const int64_t n = x.size(0);
    at::Tensor result = x.clone();

    if (n == 0) return result;

    auto r_a = result.accessor<double, 1>();
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(r_a[i])) {
            r_a[i] = fill_value;
        }
    }

    return result;
}

} // namespace xpandas
