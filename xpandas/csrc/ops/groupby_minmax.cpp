/**
 * groupby_minmax.cpp -- Groupby min, max, first, last aggregation ops.
 *
 * Ops:
 *   groupby_min(key, value)   -> (Tensor, Tensor) -- per-group minimum
 *   groupby_max(key, value)   -> (Tensor, Tensor) -- per-group maximum
 *   groupby_first(key, value) -> (Tensor, Tensor) -- per-group first value
 *   groupby_last(key, value)  -> (Tensor, Tensor) -- per-group last value
 *
 * Uses the same sorted-key + binary-search indexing pattern as groupby_agg.cpp.
 */

#include "ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace xpandas {

namespace {

struct KeyIndex {
    std::vector<int64_t> unique_keys;
    std::vector<int64_t> group_of;

    static KeyIndex build(const int64_t* keys, int64_t n) {
        KeyIndex ki;
        if (n == 0) return ki;

        std::vector<int64_t> seen;
        seen.reserve(64);
        ki.group_of.resize(n);

        for (int64_t i = 0; i < n; ++i) {
            int64_t k = keys[i];
            auto it = std::lower_bound(seen.begin(), seen.end(), k);
            if (it == seen.end() || *it != k) {
                seen.insert(it, k);
            }
        }

        ki.unique_keys = seen;

        for (int64_t i = 0; i < n; ++i) {
            auto it = std::lower_bound(seen.begin(), seen.end(), keys[i]);
            ki.group_of[i] = static_cast<int64_t>(it - seen.begin());
        }

        return ki;
    }
};

} // anonymous namespace

// ---- groupby_min ----
std::tuple<at::Tensor, at::Tensor>
groupby_min(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_min: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_min: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_min: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    constexpr double INF = std::numeric_limits<double>::infinity();
    std::vector<double> mins(ng, INF);

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = ki.group_of[i];
        if (pv[i] < mins[g]) {
            mins[g] = pv[i];
        }
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = mins[i];
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_max ----
std::tuple<at::Tensor, at::Tensor>
groupby_max(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_max: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_max: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_max: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> maxs(ng, NEG_INF);

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = ki.group_of[i];
        if (pv[i] > maxs[g]) {
            maxs[g] = pv[i];
        }
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = maxs[i];
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_first ----
std::tuple<at::Tensor, at::Tensor>
groupby_first(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_first: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_first: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_first: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> firsts(ng, NaN);
    std::vector<bool> seen(ng, false);

    for (int64_t i = 0; i < n; ++i) {
        int64_t g = ki.group_of[i];
        if (!seen[g]) {
            firsts[g] = pv[i];
            seen[g] = true;
        }
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = firsts[i];
    }

    return std::make_tuple(t_keys, t_vals);
}

// ---- groupby_last ----
std::tuple<at::Tensor, at::Tensor>
groupby_last(const at::Tensor& key, const at::Tensor& value) {
    TORCH_CHECK(key.dim() == 1, "groupby_last: key must be 1-D");
    TORCH_CHECK(value.dim() == 1, "groupby_last: value must be 1-D");
    TORCH_CHECK(key.size(0) == value.size(0),
                "groupby_last: key and value must have same length");

    const int64_t n = key.size(0);
    const int64_t* pk = key.data_ptr<int64_t>();
    const double*  pv = value.data_ptr<double>();

    auto ki = KeyIndex::build(pk, n);
    const int64_t ng = static_cast<int64_t>(ki.unique_keys.size());

    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> lasts(ng, NaN);

    // Simply overwrite — the last write wins
    for (int64_t i = 0; i < n; ++i) {
        lasts[ki.group_of[i]] = pv[i];
    }

    at::Tensor t_keys = at::empty({ng}, at::TensorOptions().dtype(at::kLong));
    at::Tensor t_vals = at::empty({ng}, at::TensorOptions().dtype(at::kDouble));
    auto* ok = t_keys.data_ptr<int64_t>();
    auto* ov = t_vals.data_ptr<double>();
    for (int64_t i = 0; i < ng; ++i) {
        ok[i] = ki.unique_keys[i];
        ov[i] = lasts[i];
    }

    return std::make_tuple(t_keys, t_vals);
}

} // namespace xpandas
