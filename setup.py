"""
setup.py -- Build the xpandas C++ extension that registers custom ops.

    pip install -e .
    # or
    python setup.py build_ext --inplace
"""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
import torch
import glob as _glob

# Collect all .cpp files under csrc/ops/
cpp_sources = sorted(_glob.glob("xpandas/csrc/ops/*.cpp"))

# Ensure ABI flag matches the installed PyTorch.
# This is critical when pip build-isolation installs a different torch
# into the temporary build environment.
_abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"

setup(
    name="xpandas",
    version="0.1.0",
    description="Pandas-like DataFrame API backed by PyTorch custom ops for TorchScript",
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name="xpandas._C",
            sources=cpp_sources,
            include_dirs=["xpandas/csrc/ops"],
            extra_compile_args=[_abi_flag],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
