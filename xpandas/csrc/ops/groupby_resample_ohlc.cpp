/**
 * groupby_resample_ohlc.cpp -- groupby + OHLC aggregation.
 *
 * Groups rows by a key column (int64 enum-encoded) and computes
 * OHLC (first / max / min / last) of a value column per group.
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace xpandas {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
groupby_resample_ohlc(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "key and value must have same length");

    const int64_t n = key.size(0);
    auto key_a   = key.accessor<int64_t, 1>();
    auto value_a = value.accessor<double, 1>();

    // Collect unique keys in order of first appearance
    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int64_t> key_to_idx;

    struct OHLC { double o, h, l, c; };
    std::vector<OHLC> stats;

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = key_a[i];
        double  v = value_a[i];
        auto it = key_to_idx.find(k);
        if (it == key_to_idx.end()) {
            int64_t idx = static_cast<int64_t>(unique_keys.size());
            key_to_idx[k] = idx;
            unique_keys.push_back(k);
            stats.push_back({v, v, v, v});
        } else {
            auto& s = stats[it->second];
            if (v > s.h) s.h = v;
            if (v < s.l) s.l = v;
            s.c = v;   // last seen
        }
    }

    int64_t ng = static_cast<int64_t>(unique_keys.size());
    auto opts_d = at::TensorOptions().dtype(at::kDouble);
    auto opts_i = at::TensorOptions().dtype(at::kLong);

    at::Tensor t_keys  = at::empty({ng}, opts_i);
    at::Tensor t_open  = at::empty({ng}, opts_d);
    at::Tensor t_high  = at::empty({ng}, opts_d);
    at::Tensor t_low   = at::empty({ng}, opts_d);
    at::Tensor t_close = at::empty({ng}, opts_d);

    auto pk = t_keys.accessor<int64_t, 1>();
    auto po = t_open.accessor<double, 1>();
    auto ph = t_high.accessor<double, 1>();
    auto pl = t_low.accessor<double, 1>();
    auto pc = t_close.accessor<double, 1>();

    for (int64_t i = 0; i < ng; ++i) {
        pk[i] = unique_keys[i];
        po[i] = stats[i].o;
        ph[i] = stats[i].h;
        pl[i] = stats[i].l;
        pc[i] = stats[i].c;
    }

    return std::make_tuple(t_keys, t_open, t_high, t_low, t_close);
}

} // namespace xpandas
