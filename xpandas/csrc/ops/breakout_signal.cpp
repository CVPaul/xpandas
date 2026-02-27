/**
 * breakout_signal.cpp -- Fused breakout signal op.
 *
 * Computes: (price > high).float() - (price < low).float()
 * Returns +1.0 (long), -1.0 (short), 0.0 (flat) per element.
 */

#include "ops.h"

namespace xpandas {

at::Tensor breakout_signal(
    const at::Tensor& price,
    const at::Tensor& high,
    const at::Tensor& low) {
    auto long_  = at::gt(price, high).to(at::kDouble);
    auto short_ = at::lt(price, low).to(at::kDouble);
    return long_ - short_;
}

} // namespace xpandas
