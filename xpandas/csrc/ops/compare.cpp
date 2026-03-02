/**
 * compare.cpp -- Element-wise comparison ops.
 */

#include "ops.h"

namespace xpandas {

at::Tensor compare_gt(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 1, "compare_gt: input 'a' must be 1-D, got ", a.dim(), "-D");
    TORCH_CHECK(b.dim() == 1, "compare_gt: input 'b' must be 1-D, got ", b.dim(), "-D");
    TORCH_CHECK(a.size(0) == b.size(0),
                "compare_gt: inputs must have same length (a=", a.size(0),
                ", b=", b.size(0), ")");
    return at::gt(a, b);
}

at::Tensor compare_lt(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.dim() == 1, "compare_lt: input 'a' must be 1-D, got ", a.dim(), "-D");
    TORCH_CHECK(b.dim() == 1, "compare_lt: input 'b' must be 1-D, got ", b.dim(), "-D");
    TORCH_CHECK(a.size(0) == b.size(0),
                "compare_lt: inputs must have same length (a=", a.size(0),
                ", b=", b.size(0), ")");
    return at::lt(a, b);
}

} // namespace xpandas
