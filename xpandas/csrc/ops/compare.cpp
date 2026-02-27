/**
 * compare.cpp -- Element-wise comparison ops.
 */

#include "ops.h"

namespace xpandas {

at::Tensor compare_gt(const at::Tensor& a, const at::Tensor& b) {
    return at::gt(a, b);
}

at::Tensor compare_lt(const at::Tensor& a, const at::Tensor& b) {
    return at::lt(a, b);
}

} // namespace xpandas
