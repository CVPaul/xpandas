/**
 * groupby_agg.cpp -- Generic groupby aggregation ops.
 *
 * Provides per-group aggregations on a 1-D value tensor, grouped by an
 * int64 key tensor.  Each op returns two tensors: (unique_keys, agg_values).
 *
 * Ops:
 *   groupby_mean(key, value)  -> (Tensor, Tensor)   -- per-group mean
 *   groupby_sum(key, value)   -> (Tensor, Tensor)   -- per-group sum
 *   groupby_count(key, value) -> (Tensor, Tensor)   -- per-group count (float64)
 *   groupby_std(key, value)   -> (Tensor, Tensor)   -- per-group std (ddof=1)
 */

#include "ops.h"

#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace xpandas {

// ---- groupby_sum ----
std::tuple<at::Tensor, at::Tensor>
groupby_sum(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_sum: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_sum: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_sum: key and value must have same length");

    const int64_t n = key.size(0);
    auto key_a   = key.accessor<int64_t, 1>();
    auto value_a = value.accessor<double, 1>();

    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int64_t> key_to_idx;
    std::vector<double> sums;

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = key_a[i];
        double  v = value_a[i];
        auto it = key_to_idx.find(k);
        if (it == key_to_idx.end()) {
            int64_t idx = static_cast<int64_t>(unique_keys.size());
            key_to_idx[k] = idx;
            unique_keys.push_back(k);
            sums.push_back(v);
        } else {
            sums[it->second] += v;
        }
    }

    int64_t ng = static_cast<int64_t>(unique_keys.size());
    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));

    auto pk = t_keys.accessor<int64_t, 1>();
    auto pv = t_vals.accessor<double, 1>();
    for (int64_t i = 0; i < ng; ++i) {
        pk[i] = unique_keys[i];
        pv[i] = sums[i];
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_mean ----
std::tuple<at::Tensor, at::Tensor>
groupby_mean(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_mean: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_mean: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_mean: key and value must have same length");

    const int64_t n = key.size(0);
    auto key_a   = key.accessor<int64_t, 1>();
    auto value_a = value.accessor<double, 1>();

    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int64_t> key_to_idx;
    std::vector<double> sums;
    std::vector<int64_t> counts;

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = key_a[i];
        double  v = value_a[i];
        auto it = key_to_idx.find(k);
        if (it == key_to_idx.end()) {
            int64_t idx = static_cast<int64_t>(unique_keys.size());
            key_to_idx[k] = idx;
            unique_keys.push_back(k);
            sums.push_back(v);
            counts.push_back(1);
        } else {
            sums[it->second] += v;
            counts[it->second] += 1;
        }
    }

    int64_t ng = static_cast<int64_t>(unique_keys.size());
    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));

    auto pk = t_keys.accessor<int64_t, 1>();
    auto pv = t_vals.accessor<double, 1>();
    for (int64_t i = 0; i < ng; ++i) {
        pk[i] = unique_keys[i];
        pv[i] = sums[i] / static_cast<double>(counts[i]);
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_count ----
std::tuple<at::Tensor, at::Tensor>
groupby_count(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_count: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_count: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_count: key and value must have same length");

    const int64_t n = key.size(0);
    auto key_a = key.accessor<int64_t, 1>();

    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int64_t> key_to_idx;
    std::vector<int64_t> counts;

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = key_a[i];
        auto it = key_to_idx.find(k);
        if (it == key_to_idx.end()) {
            int64_t idx = static_cast<int64_t>(unique_keys.size());
            key_to_idx[k] = idx;
            unique_keys.push_back(k);
            counts.push_back(1);
        } else {
            counts[it->second] += 1;
        }
    }

    int64_t ng = static_cast<int64_t>(unique_keys.size());
    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));

    auto pk = t_keys.accessor<int64_t, 1>();
    auto pv = t_vals.accessor<double, 1>();
    for (int64_t i = 0; i < ng; ++i) {
        pk[i] = unique_keys[i];
        pv[i] = static_cast<double>(counts[i]);
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_std (ddof=1, matching pandas default) ----
std::tuple<at::Tensor, at::Tensor>
groupby_std(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_std: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_std: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_std: key and value must have same length");

    const int64_t n = key.size(0);
    auto key_a   = key.accessor<int64_t, 1>();
    auto value_a = value.accessor<double, 1>();

    // Two-pass: first collect sums and counts, then compute variance.
    std::vector<int64_t> unique_keys;
    std::unordered_map<int64_t, int64_t> key_to_idx;
    std::vector<double> sums;
    std::vector<int64_t> counts;

    for (int64_t i = 0; i < n; ++i) {
        int64_t k = key_a[i];
        double  v = value_a[i];
        auto it = key_to_idx.find(k);
        if (it == key_to_idx.end()) {
            int64_t idx = static_cast<int64_t>(unique_keys.size());
            key_to_idx[k] = idx;
            unique_keys.push_back(k);
            sums.push_back(v);
            counts.push_back(1);
        } else {
            sums[it->second] += v;
            counts[it->second] += 1;
        }
    }

    int64_t ng = static_cast<int64_t>(unique_keys.size());

    // Compute means
    std::vector<double> means(ng);
    for (int64_t i = 0; i < ng; ++i) {
        means[i] = sums[i] / static_cast<double>(counts[i]);
    }

    // Second pass: compute sum of squared deviations
    std::vector<double> sq_devs(ng, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        int64_t idx = key_to_idx[key_a[i]];
        double diff = value_a[i] - means[idx];
        sq_devs[idx] += diff * diff;
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));

    auto pk = t_keys.accessor<int64_t, 1>();
    auto pv = t_vals.accessor<double, 1>();
    for (int64_t i = 0; i < ng; ++i) {
        pk[i] = unique_keys[i];
        if (counts[i] <= 1) {
            pv[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            pv[i] = std::sqrt(sq_devs[i] / static_cast<double>(counts[i] - 1));
        }
    }

    return std::make_tuple(t_keys, t_vals);
}

} // namespace xpandas
