cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(xpandas_inference CXX)

# Find Torch (libtorch).
# Pass -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect all C++ op source files
file(GLOB XPANDAS_OPS_SOURCES xpandas/csrc/ops/*.cpp)

# --------------------------------------------------------------------------
# 1. xpandas_ops -- shared library containing the custom ops.
#    When loaded (via dlopen or linked), the TORCH_LIBRARY static
#    initialisers register xpandas::* in the PyTorch dispatcher.
# --------------------------------------------------------------------------
add_library(xpandas_ops SHARED ${XPANDAS_OPS_SOURCES})
target_link_libraries(xpandas_ops "${TORCH_LIBRARIES}")
target_include_directories(xpandas_ops PRIVATE
    "${TORCH_INCLUDE_DIRS}"
    "xpandas/csrc/ops"
)

# --------------------------------------------------------------------------
# 2. alpha_infer -- the pure-C++ inference binary.
# --------------------------------------------------------------------------
add_executable(alpha_infer inference/main.cpp)
target_link_libraries(alpha_infer "${TORCH_LIBRARIES}" dl)
target_include_directories(alpha_infer PRIVATE "${TORCH_INCLUDE_DIRS}")

# Copy the built library next to the executable for convenience
add_custom_command(TARGET xpandas_ops POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:xpandas_ops>
        ${CMAKE_BINARY_DIR}/$<TARGET_FILE_NAME:xpandas_ops>
)
