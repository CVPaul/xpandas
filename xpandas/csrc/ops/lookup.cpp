/**
 * lookup.cpp -- Dict[str, Tensor] column lookup.
 *
 * Equivalent to df['column_name'] -- needed because TorchScript
 * doesn't allow __getattr__ dispatch on Dict.
 */

#include "ops.h"

namespace xpandas {

at::Tensor lookup(
    const c10::Dict<std::string, at::Tensor>& table,
    const std::string& key) {
    TORCH_CHECK(table.contains(key),
                "lookup: key '", key, "' not found in DataFrame. ",
                "Check column names and spelling.");
    return table.at(key);
}

} // namespace xpandas
