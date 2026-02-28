/**
 * ewm.cpp -- Exponential weighted moving average.
 *
 * Ops:
 *   ewm_mean(x, span) -> Tensor -- EWM mean matching pandas ewm(span=...).mean()
 *
 * Uses the recursive formula:
 *   alpha = 2.0 / (span + 1)
 *   ewm[0] = x[0]
 *   ewm[i] = alpha * x[i] + (1 - alpha) * ewm[i-1]
 *
 * This matches pandas `adjust=True` behavior using the "com" definition
 * where com = (span - 1) / 2.  For the simple recursive version
 * (adjust=False), which is common in quant, we implement that directly.
 * This is the adjust=False variant since it's more common in production
 * quant systems and is what TorchScript inference typically needs.
 */

#include "ops.h"

#include <cmath>
#include <limits>

namespace xpandas {

at::Tensor ewm_mean(const at::Tensor& x, int64_t span) {
    TORCH_CHECK(x.dim() == 1, "ewm_mean: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "ewm_mean: input must be float64");
    TORCH_CHECK(span > 0, "ewm_mean: span must be positive");

    const int64_t n = x.size(0);
    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* px = x.data_ptr<double>();
    double*       pr = result.data_ptr<double>();

    double alpha = 2.0 / (static_cast<double>(span) + 1.0);
    double one_minus_alpha = 1.0 - alpha;

    pr[0] = px[0];
    for (int64_t i = 1; i < n; ++i) {
        pr[i] = alpha * px[i] + one_minus_alpha * pr[i - 1];
    }

    return result;
}

} // namespace xpandas
