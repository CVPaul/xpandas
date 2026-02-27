/**
 * clip.cpp -- Value clipping.
 *
 * Equivalent to pandas Series.clip(lower, upper).
 * Clips values to [lower, upper] range.
 */

#include "ops.h"

namespace xpandas {

at::Tensor clip(const at::Tensor& x, double lower, double upper) {
    TORCH_CHECK(x.dim() == 1, "clip: input must be 1-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "clip: input must be float64");
    TORCH_CHECK(lower <= upper,
                "clip: lower (", lower, ") must be <= upper (", upper, ")");

    return at::clamp(x, lower, upper);
}

} // namespace xpandas
