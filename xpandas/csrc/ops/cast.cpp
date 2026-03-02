/**
 * cast.cpp -- Type-casting ops.
 */

#include "ops.h"

namespace xpandas {

at::Tensor bool_to_float(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "bool_to_float: input must be 1-D, got ", x.dim(), "-D");
    return x.to(at::kDouble);
}

} // namespace xpandas
