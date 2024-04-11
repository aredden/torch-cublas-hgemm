import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

generator_flag = []
torch_dir = torch.__path__[0]


cc_flag = []


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "8"
    return nvcc_extra_args + ["--threads", nvcc_threads]


setup(
    name="cublas_ops",
    version="0.0.2",
    ext_modules=[
        CUDAExtension(
            "cublas_ops_ext",
            [
                "src/cublas_hgemm.cpp",
                "src/cublas_hgemm_kernel.cu",
                "src/cublas_hgemm_batched_kernel.cu",
                "src/cublaslt_hgemm_kernel.cu",
                "src/cublaslt_hgemm_batched_kernel.cu",
                "src/cudnn_convolution_kernel.cpp",
                # 'src/cublas_state.cpp'
            ],
            extra_link_flags=["-lcudnn"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + generator_flag,
                "nvcc": append_nvcc_threads(
                    [
                        "-O3",
                        "-std=c++17",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_HALF2_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--use_fast_math",
                    ]
                    + generator_flag
                    + cc_flag
                ),
            },
        ),
    ],
    packages=find_packages(),
    cmdclass={"build_ext": BuildExtension},
)
