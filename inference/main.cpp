/**
 * inference/main.cpp -- Pure C++ inference driver for the Alpha model.
 *
 * Build:
 *   mkdir build && cd build
 *   cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
 *   make -j
 *
 * Run:
 *   ./alpha_infer  alpha.pt  libxpandas_ops.so
 *
 * The program:
 *   1. Loads the xpandas custom-ops shared library
 *   2. Loads the TorchScript model
 *   3. Constructs sample market data as Dict[str, Tensor]
 *   4. Calls on_bod() then forward() and prints the signal
 */

#include <torch/script.h>
#include <torch/torch.h>

#include <dlfcn.h>
#include <iostream>
#include <string>

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.pt> <libxpandas_ops.so>\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ops_lib    = argv[2];

    // ---------------------------------------------------------------
    // 1. Load the xpandas custom-ops .so via dlopen.
    //    This triggers the TORCH_LIBRARY / TORCH_LIBRARY_IMPL static
    //    initializers, registering xpandas::* ops in the dispatcher.
    // ---------------------------------------------------------------
    void* handle = dlopen(ops_lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::cerr << "Failed to load ops library: " << dlerror() << "\n";
        return 1;
    }
    std::cout << "[OK] Loaded custom ops from: " << ops_lib << "\n";

    // ---------------------------------------------------------------
    // 2. Load the TorchScript model
    // ---------------------------------------------------------------
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[OK] Model loaded from: " << model_path << "\n";

    // ---------------------------------------------------------------
    // 3. Construct sample market data (Dict[str, Tensor])
    // ---------------------------------------------------------------
    // Simulate 6 ticks: 2 instruments x 3 ticks each
    //   InstrumentID: [0, 0, 0, 1, 1, 1]   (enum-encoded)
    //   price:        [100, 105, 102, 200, 198, 210]

    auto inst_key = torch::tensor({0L, 0L, 0L, 1L, 1L, 1L}, torch::kLong);
    auto price    = torch::tensor({100.0, 105.0, 102.0, 200.0, 198.0, 210.0},
                                  torch::kDouble);

    c10::Dict<std::string, at::Tensor> hist_data;
    hist_data.insert("InstrumentID", inst_key);
    hist_data.insert("price", price);

    // --- Call on_bod ---
    int64_t timestamp = 1700000000;
    std::vector<torch::jit::IValue> bod_args;
    bod_args.push_back(timestamp);
    bod_args.push_back(hist_data);
    module.get_method("on_bod")(bod_args);
    std::cout << "[OK] on_bod() done.\n";

    // ---------------------------------------------------------------
    // 4. Call forward with current tick data
    // ---------------------------------------------------------------
    // inst0 price=106 (breaks above high=105) -> +1
    // inst1 price=195 (breaks below low=198)  -> -1
    auto current_price = torch::tensor({106.0, 195.0}, torch::kDouble);

    c10::Dict<std::string, at::Tensor> tick_data;
    tick_data.insert("price", current_price);

    std::vector<torch::jit::IValue> fwd_args;
    fwd_args.push_back(timestamp);
    fwd_args.push_back(tick_data);

    auto result = module.forward(fwd_args);
    auto signal = result.toTensor();

    std::cout << "[OK] Signal: " << signal << "\n";
    std::cout << "Expected: [+1.0, -1.0]\n";

    return 0;
}
