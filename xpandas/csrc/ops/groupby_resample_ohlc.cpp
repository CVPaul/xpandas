/**
 * groupby_resample_ohlc.cpp -- groupby + OHLC aggregation.
 *
 * Groups rows by a key column (int64 enum-encoded) and computes
 * OHLC (first / max / min / last) of a value column per group.
 *
 * Optimized: uses raw data pointers and sorted-key + binary search
 * instead of std::unordered_map.
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace xpandas {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
groupby_resample_ohlc(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    // Build sorted unique keys and group indices
    std::vector<int64_t> sorted_keys;
    sorted_keys.reserve(64);
    std::vector<int64_t> group_of(n);

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = pk[i];
        auto it = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), k);
        if (it == sorted_keys.end() || *it != k) {
            sorted_keys.insert(it, k);
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        auto it = std::lower_bound(sorted_keys.begin(), sorted_keys.end(), pk[i]);
        group_of[i] = static_cast<int64_t>(it - sorted_keys.begin());
    }

    const int64_t ng = static_cast<int64_t>(sorted_keys.size());

    struct OHLC { double o, h, l, c; bool init; };
    std::vector<OHLC> stats(ng, {0.0, 0.0, 0.0, 0.0, false});

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = group_of[i];
        double  v = pv[i];
        auto& s = stats[g];
        if (!s.init) {
            s = {v, v, v, v, true};
        } else {
            if (v > s.h) s.h = v;
            if (v < s.l) s.l = v;
            s.c = v;
        }
    }

    auto opts_d = at::TensorOptions().dtype(at::kDouble);
    auto opts_i = at::TensorOptions().dtype(at::kLong);

    at::Tensor t_keys  = at::empty({ng}, opts_i);
    at::Tensor t_open  = at::empty({ng}, opts_d);
    at::Tensor t_high  = at::empty({ng}, opts_d);
    at::Tensor t_low   = at::empty({ng}, opts_d);
    at::Tensor t_close = at::empty({ng}, opts_d);

    auto* ok = t_keys.data_ptr<int64_t>();
    auto* oo = t_open.data_ptr<double>();
    auto* oh = t_high.data_ptr<double>();
    auto* ol = t_low.data_ptr<double>();
    auto* oc = t_close.data_ptr<double>();

    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = sorted_keys[i];
        oo[i] = stats[i].o;
        oh[i] = stats[i].h;
        ol[i] = stats[i].l;
        oc[i] = stats[i].c;
    }

    return std::make_tuple(t_keys, t_open, t_high, t_low, t_close);
}

} // namespace xpandas
