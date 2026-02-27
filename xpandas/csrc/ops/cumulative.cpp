/**
 * cumulative.cpp -- Cumulative operations.
 *
 * Ops:
 *   cumsum(x)  -> Tensor   -- cumulative sum (NaN-aware: NaN propagates)
 *   cumprod(x) -> Tensor   -- cumulative product (NaN-aware: NaN propagates)
 *
 * Matches pandas Series.cumsum() / Series.cumprod() behavior.
 */

#include "ops.h"

#include <cmath>

namespace xpandas {

at::Tensor cumsum(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "cumsum: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "cumsum: input must be float64");

    // Use torch::cumsum which handles the computation efficiently
    return at::cumsum(x, /*dim=*/0);
}

at::Tensor cumprod(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "cumprod: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "cumprod: input must be float64");

    return at::cumprod(x, /*dim=*/0);
}

} // namespace xpandas
