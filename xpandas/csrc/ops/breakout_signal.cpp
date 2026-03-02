/**
 * breakout_signal.cpp -- Fused breakout signal op.
 *
 * Computes: (price > high).float() - (price < low).float()
 * Returns +1.0 (long), -1.0 (short), 0.0 (flat) per element.
 *
 * Optimized: single pass over raw pointers, no intermediate tensors.
 */

#include "ops.h"

namespace xpandas {

at::Tensor breakout_signal(
    const at::Tensor& price,
    const at::Tensor& high,
    const at::Tensor& low) {
    TORCH_CHECK(price.dim() == 1 && high.dim() == 1 && low.dim() == 1,
                "breakout_signal: all inputs must be 1-D");
    TORCH_CHECK(price.scalar_type() == at::kDouble &&
                high.scalar_type() == at::kDouble &&
                low.scalar_type() == at::kDouble,
                "breakout_signal: all inputs must be float64 (Double)");
    const int64_t n = price.size(0);
    TORCH_CHECK(high.size(0) == n && low.size(0) == n,
                "breakout_signal: all inputs must have same length");

    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));

    if (n == 0) return result;

    const double* p_price = price.data_ptr<double>();
    const double* p_high  = high.data_ptr<double>();
    const double* p_low   = low.data_ptr<double>();
    double*       p_out   = result.data_ptr<double>();

    for (int64_t i = 0; i < n; ++i) {
        double sig = 0.0;
        if (p_price[i] > p_high[i]) sig = 1.0;
        else if (p_price[i] < p_low[i]) sig = -1.0;
        p_out[i] = sig;
    }

    return result;
}

} // namespace xpandas
