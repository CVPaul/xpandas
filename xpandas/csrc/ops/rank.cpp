/**
 * rank.cpp -- Per-element rank within a 1-D tensor.
 *
 * Implements pandas-style Series.rank(method='average', na_option='keep').
 *
 * For a 1-D float64 input of length N, returns a float64 tensor where
 * each element is its 1-based rank (average of ties). NaN values in
 * the input produce NaN in the output.
 *
 * Example:
 *   input:  [3.0, 1.0, 2.0, 1.0]
 *   output: [4.0, 1.5, 3.0, 1.5]
 *
 * This op was added as a walkthrough example in CONTRIBUTING.md.
 *
 * Optimized: uses raw data pointers instead of accessors for better
 * performance.
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace xpandas {

at::Tensor rank(const at::Tensor& x) {
    TORCH_CHECK(x.dim() == 1, "rank: input must be 1-D, got ", x.dim(), "-D");
    TORCH_CHECK(x.scalar_type() == at::kDouble,
                "rank: input must be float64 (Double)");

    const int64_t n = x.size(0);
    if (n == 0) {
        return at::empty({0}, at::TensorOptions().dtype(at::kDouble));
    }

    const double* px = x.data_ptr<double>();

    // Build an index array and sort by value (NaN goes to end).
    std::vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [px](int64_t a, int64_t b) {
        double va = px[a], vb = px[b];
        // NaN sorts to the end
        if (std::isnan(va)) return false;
        if (std::isnan(vb)) return true;
        return va < vb;
    });

    at::Tensor result = at::empty({n}, at::TensorOptions().dtype(at::kDouble));
    double* pr = result.data_ptr<double>();

    // Walk through sorted indices and assign average ranks to ties.
    int64_t i = 0;
    while (i < n) {
        double val = px[idx[i]];

        // NaN values get NaN rank
        if (std::isnan(val)) {
            for (; i < n; ++i) {
                pr[idx[i]] = std::numeric_limits<double>::quiet_NaN();
            }
            break;
        }

        // Find the run of identical values (ties)
        int64_t j = i + 1;
        while (j < n && !std::isnan(px[idx[j]]) && px[idx[j]] == val) {
            ++j;
        }

        // Average rank for this tie group: 1-based ranks from (i+1) to j
        double avg_rank = 0.5 * (static_cast<double>(i + 1) +
                                  static_cast<double>(j));
        for (int64_t k = i; k < j; ++k) {
            pr[idx[k]] = avg_rank;
        }

        i = j;
    }

    return result;
}

} // namespace xpandas
