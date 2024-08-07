// inspired by https://github.com/wangsiping97/FastGEMV
// but with half precision accumulate and 1x8 by 1x8 dot via __hfma2 and __hmul multiplication
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

__device__ __forceinline__ half warpReduceSum(half sum, uint32_t threadNum)
{
    if (threadNum >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (threadNum >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8); // 0-8, 1-9, 2-10, etc.
    if (threadNum >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4); // 0-4, 1-5, 2-6, etc.
    if (threadNum >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2); // 0-2, 1-3, 4-6, 5-7, etc.
    if (threadNum >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1); // 0-1, 2-3, 4-5, etc.
    return sum;
}

// Multiply 8 half precision numbers by 8 other half precision numbers and accumulate the result on a single thread.
__device__ __forceinline__ half hmul2_4half2(const half2 *a, const half2 *b)
{
    half2 res = __hfma2(a[3], b[3], __hfma2(a[2], b[2], __hfma2(a[1], b[1], __hmul2(a[0], b[0]))));
    return res.x + res.y;
}

// inspired by https://github.com/wangsiping97/FastGEMV/blob/main/fast_gemv.cu
__global__ void
simt_hgemv_op_kernel(const half *mat, const half *vec, half *res, uint32_t n, uint32_t num_per_thread)
{
    half sum = 0.f;
    // each thread load num_per_thread elements from global
    uint32_t tid = threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t start_idx = threadIdx.x;
    const float4 *mat4 = reinterpret_cast<const float4 *>(mat);
    const float4 *vec4 = reinterpret_cast<const float4 *>(vec);

#pragma unroll
    for (int32_t iter = 0; iter < num_per_thread >> 3; iter++)
    {
        uint32_t j = start_idx + iter * blockDim.x;
        if (j < n >> 3)
        {
            float4 vec_val = vec4[j];
            float4 mat_val = mat4[row * (n >> 3) + j];
            sum += hmul2_4half2(reinterpret_cast<const half2 *>(&vec_val), reinterpret_cast<const half2 *>(&mat_val));
        }
    }

    sum = warpReduceSum(sum, blockDim.x);

    if (blockDim.x <= WARP_SIZE)
    {
        if (tid == 0)
        {
            res[row] = sum;
        }
        return;
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ half warpLevelSums[SHARED_MEM_MAX_ROWS][WARP_SIZE];
    const int32_t laneId = threadIdx.x % WARP_SIZE;
    const int32_t warpId = threadIdx.x / WARP_SIZE;
    if (laneId == 0)
        warpLevelSums[threadIdx.y][warpId] = sum;
    __syncthreads();
    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[threadIdx.y][laneId] : half(0.0f);
    // Final reduce using first warp
    if (warpId == 0)
        sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
    if (tid == 0)
    {
        res[row] = sum;
    }
}

struct PsudoTensor
{
    uint32_t height_;
    uint32_t width_;
};

#define CIDV_CALC

torch::Tensor simt_hgemv_op(
    const torch::Tensor mat, const torch::Tensor vec, uint32_t block_dim_x = 32, uint32_t block_dim_y = 4)
{
    at::DeviceGuard guard(mat.device());

    PsudoTensor mat_psudo = {uint32_t(mat.size(0)), uint32_t(mat.size(1))};
    PsudoTensor vec_psudo = {uint32_t(vec.size(0)), uint32_t(vec.size(1))};
    TORCH_CHECK(mat_psudo.width_ == vec_psudo.height_, "Matrix and vector dimensions do not match");
    TORCH_CHECK(block_dim_y <= SHARED_MEM_MAX_ROWS);
    TORCH_CHECK(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
    uint32_t num_per_thread = mat_psudo.width_ / block_dim_x;
    TORCH_CHECK(num_per_thread >= 8, "Matrix is too small, try reducing block_dim_x (via block_dim_x argument)");

    torch::Tensor result_tensor = torch::empty({mat_psudo.height_, 1}, mat.options());

    dim3 grid_dim(1, (mat_psudo.height_ + 1) / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    simt_hgemv_op_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<const half *>(mat.const_data_ptr()),
        reinterpret_cast<const half *>(vec.const_data_ptr()),
        reinterpret_cast<half *>(result_tensor.mutable_data_ptr<at::Half>()),
        mat_psudo.width_,
        num_per_thread);
    checkCudaStatus(cudaPeekAtLastError());
    return result_tensor;
}

void initGemvCustomBindings(py::module &m)
{
    m.def(
        "_simt_hgemv",
        py::overload_cast<torch::Tensor, torch::Tensor, uint32_t, uint32_t>(&simt_hgemv_op),
        py::arg("mat"),
        py::arg("vec"),
        py::arg("block_dim_x") = 32,
        py::arg("block_dim_y") = 4,
        "simt_gemv fp16 accumulate gemv");
}