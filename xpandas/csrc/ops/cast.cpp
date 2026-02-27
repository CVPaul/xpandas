/**
 * cast.cpp -- Type-casting ops.
 */

#include "ops.h"

namespace xpandas {

at::Tensor bool_to_float(const at::Tensor& x) {
    return x.to(at::kDouble);
}

} // namespace xpandas
