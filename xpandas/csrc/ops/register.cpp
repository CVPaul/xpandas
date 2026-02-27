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
}

TORCH_LIBRARY_IMPL(xpandas, CPU, m) {
    m.impl("groupby_resample_ohlc", &xpandas::groupby_resample_ohlc);
    m.impl("compare_gt",            &xpandas::compare_gt);
    m.impl("compare_lt",            &xpandas::compare_lt);
    m.impl("bool_to_float",         &xpandas::bool_to_float);
    // lookup is registered as catch-all in TORCH_LIBRARY above
    m.impl("breakout_signal",       &xpandas::breakout_signal);
    m.impl("rank",                  &xpandas::rank);
    m.impl("to_datetime",           &xpandas::to_datetime);
    m.impl("dt_floor",              &xpandas::dt_floor);
}
