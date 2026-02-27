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
 *
 * Optimized: uses raw data pointers and flat-array indexing (via sorted unique
 * keys + binary search) instead of std::unordered_map for lower overhead.
 * groupby_std uses Welford's online algorithm in a single pass.
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xpandas {

namespace {

// Build a sorted array of unique keys from an int64 key tensor, and return
// a mapping from each element's key to a dense group index [0, num_groups).
// This replaces std::unordered_map for better cache behavior and lower overhead.
struct KeyIndex {
    std::vector<int64_t> unique_keys;  // sorted unique keys
    std::vector<int64_t> group_of;     // group_of[i] = group index for element i

    static KeyIndex build(const int64_t* keys, int64_t n) {
        KeyIndex ki;
        if (n == 0) return ki;

        // Collect unique keys, preserving first-appearance order is not needed
        // since we sort them — but for compatibility with existing tests we
        // maintain insertion order.  Use a small sorted vector + binary search.
        // First pass: collect all unique keys in appearance order.
        std::vector<int64_t> seen;
        seen.reserve(64);
        ki.group_of.resize(n);

        for (int64_t i = 0; i < n; ++i) {
            int64_t k = keys[i];
            // Binary search in the sorted `seen` array
            auto it = std::lower_bound(seen.begin(), seen.end(), k);
            if (it == seen.end() || *it != k) {
                seen.insert(it, k);
            }
        }

        ki.unique_keys = seen;  // sorted

        // Second pass: assign group indices via binary search
        const int64_t ng = static_cast<int64_t>(seen.size());
        for (int64_t i = 0; i < n; ++i) {
            auto it = std::lower_bound(seen.begin(), seen.end(), keys[i]);
            ki.group_of[i] = static_cast<int64_t>(it - seen.begin());
        }

        return ki;
    }
};

} // anonymous namespace

// ---- groupby_sum ----
std::tuple<at::Tensor, at::Tensor>
groupby_sum(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_sum: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_sum: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_sum: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    std::vector<double> sums(ng, 0.0);
    for (int64_t i = 0; i < n; ++i) {
        sums[ki.group_of[i]] += pv[i];
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = sums[i];
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
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    std::vector<double> sums(ng, 0.0);
    std::vector<int64_t> counts(ng, 0);

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = ki.group_of[i];
        sums[g]   += pv[i];
        counts[g] += 1;
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = sums[i] / static_cast<double>(counts[i]);
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
    const int64_t* pk = key.data_ptr<int64_t>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    std::vector<int64_t> counts(ng, 0);
    for (int64_t i = 0; i < n; ++i) {
        counts[ki.group_of[i]] += 1;
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = static_cast<double>(counts[i]);
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_std (ddof=1, matching pandas default) ----
// Uses Welford's online algorithm: single pass, numerically stable.
std::tuple<at::Tensor, at::Tensor>
groupby_std(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_std: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_std: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_std: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    // Welford accumulators per group
    std::vector<int64_t> counts(ng, 0);
    std::vector<double>  means(ng, 0.0);
    std::vector<double>  m2s(ng, 0.0);

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = ki.group_of[i];
        double x = pv[i];
        counts[g] += 1;
        double delta = x - means[g];
        means[g] += delta / static_cast<double>(counts[g]);
        double delta2 = x - means[g];
        m2s[g] += delta * delta2;
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        if (counts[i] <= 1) {
            ov[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
            ov[i] = std::sqrt(m2s[i] / static_cast<double>(counts[i] - 1));
        }
    }

    return std::make_tuple(t_keys, t_vals);
}

} // namespace xpandas
