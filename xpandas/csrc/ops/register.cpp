/**
 * register.cpp -- TORCH_LIBRARY schema definitions + CPU implementations.
 *
 * This file ties together all individual op implementations and registers
 * them in the PyTorch dispatcher under the "xpandas" namespace.
 *
 * To add a new op:
 *   1. Declare it in ops.h
 *   2. Implement it in a new .cpp file
 *   3. Add a m.def() schema here
 *   4. Add a m.impl() binding here
 *   See CONTRIBUTING.md for a step-by-step walkthrough.
 */

#include "ops.h"

TORCH_LIBRARY(xpandas, m) {
    // --- groupby / aggregation ---
    m.def("groupby_resample_ohlc(Tensor key, Tensor value)"
          " -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("groupby_sum(Tensor key, Tensor value) -> (Tensor, Tensor)");
    m.def("groupby_mean(Tensor key, Tensor value) -> (Tensor, Tensor)");
    m.def("groupby_count(Tensor key, Tensor value) -> (Tensor, Tensor)");
    m.def("groupby_std(Tensor key, Tensor value) -> (Tensor, Tensor)");

    // --- element-wise comparison ---
    m.def("compare_gt(Tensor a, Tensor b) -> Tensor");
    m.def("compare_lt(Tensor a, Tensor b) -> Tensor");

    // --- type casting ---
    m.def("bool_to_float(Tensor x) -> Tensor");

    // --- dataframe utilities ---
    // NOTE: lookup has no Tensor positional args, so the dispatcher cannot
    // infer a dispatch key.  We register it as a catch-all by passing the
    // implementation directly to m.def() (CompositeImplicitAutograd).
    m.def("lookup(Dict(str, Tensor) table, str key) -> Tensor",
          &xpandas::lookup);

    // --- fused signals ---
    m.def("breakout_signal(Tensor price, Tensor high, Tensor low) -> Tensor");

    // --- statistical ---
    m.def("rank(Tensor x) -> Tensor");

    // --- datetime ---
    m.def("to_datetime(Tensor epochs, str unit) -> Tensor");
    m.def("dt_floor(Tensor dt_ns, int interval_ns) -> Tensor");

    // --- rolling window ---
    m.def("rolling_sum(Tensor x, int window) -> Tensor");
    m.def("rolling_mean(Tensor x, int window) -> Tensor");
    m.def("rolling_std(Tensor x, int window) -> Tensor");

    // --- shift / lag ---
    m.def("shift(Tensor x, int periods) -> Tensor");

    // --- NaN handling ---
    m.def("fillna(Tensor x, float fill_value) -> Tensor");

    // --- conditional ---
    m.def("where_(Tensor cond, Tensor x, Tensor other) -> Tensor");
    m.def("masked_fill(Tensor x, Tensor mask, float fill_value) -> Tensor");

    // --- percentage change ---
    m.def("pct_change(Tensor x, int periods) -> Tensor");

    // --- cumulative ---
    m.def("cumsum(Tensor x) -> Tensor");
    m.def("cumprod(Tensor x) -> Tensor");

    // --- clipping ---
    m.def("clip(Tensor x, float lower, float upper) -> Tensor");
}

TORCH_LIBRARY_IMPL(xpandas, CPU, m) {
    m.impl("groupby_resample_ohlc", &xpandas::groupby_resample_ohlc);
    m.impl("groupby_sum",           &xpandas::groupby_sum);
    m.impl("groupby_mean",          &xpandas::groupby_mean);
    m.impl("groupby_count",         &xpandas::groupby_count);
    m.impl("groupby_std",           &xpandas::groupby_std);
    m.impl("compare_gt",            &xpandas::compare_gt);
    m.impl("compare_lt",            &xpandas::compare_lt);
    m.impl("bool_to_float",         &xpandas::bool_to_float);
    // lookup is registered as catch-all in TORCH_LIBRARY above
    m.impl("breakout_signal",       &xpandas::breakout_signal);
    m.impl("rank",                  &xpandas::rank);
    m.impl("to_datetime",           &xpandas::to_datetime);
    m.impl("dt_floor",              &xpandas::dt_floor);
    m.impl("rolling_sum",           &xpandas::rolling_sum);
    m.impl("rolling_mean",          &xpandas::rolling_mean);
    m.impl("rolling_std",           &xpandas::rolling_std);
    m.impl("shift",                 &xpandas::shift);
    m.impl("fillna",                &xpandas::fillna);
    m.impl("where_",                &xpandas::where_);
    m.impl("masked_fill",           &xpandas::masked_fill);
    m.impl("pct_change",            &xpandas::pct_change);
    m.impl("cumsum",                &xpandas::cumsum);
    m.impl("cumprod",               &xpandas::cumprod);
    m.impl("clip",                  &xpandas::clip);
}
