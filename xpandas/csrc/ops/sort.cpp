/**
 * sort.cpp -- Sorting operations on Dict[str, Tensor] DataFrames.
 *
 * Ops:
 *   sort_by(Dict(str, Tensor) table, str by, bool ascending) -> Dict(str, Tensor)
 *     -- Sort all columns by the values in column `by`.
 *        Equivalent to pandas `df.sort_values(by=col, ascending=ascending)`.
 *
 * NOTE: This op has no Tensor positional args (first arg is Dict), so it must
 * be registered as catch-all (CompositeImplicitAutograd) like `lookup`.
 */

#include "ops.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace xpandas {

c10::Dict<std::string, at::Tensor>
sort_by(const c10::Dict<std::string, at::Tensor>& table,
        const std::string& by,
        bool ascending) {
    TORCH_CHECK(table.contains(by),
                "sort_by: column '", by, "' not found in table");

    const at::Tensor& sort_col = table.at(by);
    TORCH_CHECK(sort_col.dim() == 1, "sort_by: sort column must be 1-D");

    const int64_t n = sort_col.size(0);

    // Build argsort indices
    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    if (sort_col.scalar_type() == at::kDouble) {
        const double* data = sort_col.data_ptr<double>();
        if (ascending) {
            std::stable_sort(indices.begin(), indices.end(),
                [data](int64_t a, int64_t b) { return data[a] < data[b]; });
        } else {
            std::stable_sort(indices.begin(), indices.end(),
                [data](int64_t a, int64_t b) { return data[a] > data[b]; });
        }
    } else if (sort_col.scalar_type() == at::kLong) {
        const int64_t* data = sort_col.data_ptr<int64_t>();
        if (ascending) {
            std::stable_sort(indices.begin(), indices.end(),
                [data](int64_t a, int64_t b) { return data[a] < data[b]; });
        } else {
            std::stable_sort(indices.begin(), indices.end(),
                [data](int64_t a, int64_t b) { return data[a] > data[b]; });
        }
    } else {
        TORCH_CHECK(false, "sort_by: unsupported dtype (expected float64 or int64)");
    }

    // Build index tensor for gather
    at::Tensor idx = at::empty({n}, at::TensorOptions().dtype(at::kLong));
    auto* pidx = idx.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        pidx[i] = indices[i];
    }

    // Apply the permutation to every column
    c10::Dict<std::string, at::Tensor> result;
    for (auto it = table.begin(); it != table.end(); ++it) {
        const std::string& col_name = it->key();
        const at::Tensor& col = it->value();
        TORCH_CHECK(col.dim() == 1 && col.size(0) == n,
                    "sort_by: all columns must be 1-D with same length");
        result.insert(col_name, col.index_select(0, idx));
    }

    return result;
}

} // namespace xpandas
