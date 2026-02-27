/**
 * where.cpp -- Conditional selection.
 *
 * Equivalent to pandas Series.where(cond, other).
 * Returns x where cond is true, other where cond is false.
 *
 * Also: masked_fill: returns x where cond is true, fill_value otherwise.
 * (Equivalent to np.where(cond, x, fill_value))
 */

#include "ops.h"

#include <cmath>

namespace xpandas {

at::Tensor where_(const at::Tensor& cond, const at::Tensor& x,
                  const at::Tensor& other) {
    TORCH_CHECK(cond.dim() == 1, "where_: cond must be 1-D");
    TORCH_CHECK(x.dim() == 1, "where_: x must be 1-D");
    TORCH_CHECK(other.dim() == 1, "where_: other must be 1-D");
    TORCH_CHECK(cond.size(0) == x.size(0) && x.size(0) == other.size(0),
                "where_: all inputs must have same length");
    TORCH_CHECK(cond.scalar_type() == at::kBool,
                "where_: cond must be bool");

    return at::where(cond, x, other);
}

at::Tensor masked_fill(const at::Tensor& x, const at::Tensor& mask,
                       double fill_value) {
    TORCH_CHECK(x.dim() == 1, "masked_fill: x must be 1-D");
    TORCH_CHECK(mask.dim() == 1, "masked_fill: mask must be 1-D");
    TORCH_CHECK(x.size(0) == mask.size(0),
                "masked_fill: x and mask must have same length");
    TORCH_CHECK(mask.scalar_type() == at::kBool,
                "masked_fill: mask must be bool");

    // mask=True means "fill", so we return: where(!mask, x, fill_value)
    return x.masked_fill(mask, fill_value);
}

} // namespace xpandas
