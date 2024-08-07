#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string.h>

inline void checkCudaStatus(cudaError_t status)
{
    if (status != cudaSuccess)
    {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

torch::Tensor cublas_hgemm_batched_kernel(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    bool trans_a = true,
    bool trans_b = false,
    int lda = -1,
    int ldb = -1,
    int ldc = -1,
    int out_h = -1,
    int out_w = -1,
    bool weight_is_a = true)
{
    c10::cuda::CUDAGuard device_guard(a.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    half alpha = 1.0f;
    half beta = 0.0f;

    size_t batch_sz;
    if (a.ndimension() == 3)
    {
        batch_sz = a.size(0);
    }
    else if (b.ndimension() == 3)
    {
        batch_sz = b.size(0);
    }
    else
    {
        throw std::logic_error("batch size not found, either a or b should be 3D tensor!");
    }
    long long stride_a = a.stride(0);
    long long stride_b = b.stride(0);
    if (a.ndimension() == 3)
    {
        stride_a = a.stride(0);
    }
    else
    {
        stride_a = 0;
    }
    if (b.ndimension() == 3)
    {
        stride_b = b.stride(0);
    }
    else
    {
        stride_b = 0;
    }

    if (out_h == -1)
    {
        out_h = m;
    }

    if (out_w == -1)
    {
        out_w = n;
    }

    at::Tensor out = at::empty({int(batch_sz), out_h, out_w}, a.options().device(a.device()));
    long long stride_c = out.stride(0);

    cublasOperation_t OP_A = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t OP_B = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    if (lda == -1)
    {
        lda = OP_A == CUBLAS_OP_N ? m : k;
    }
    if (ldb == -1)
    {
        ldb = OP_B == CUBLAS_OP_N ? k : n;
    }
    if (ldc == -1)
    {
        ldc = m;
    }
    /*
    cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
                                    cublasOperation_t transa,
                                    cublasOperation_t transb,
                                    int m, int n, int k,
                                    const __half          *alpha,
                                    const __half          *const Aarray[], int lda,
                                    size_t strideA,
                                    const __half          *const Barray[], int ldb,
                                    size_t strideB,
                                    const __half          *beta,
                                    __half          *const Carray[], int ldc,
                                    size_t strideC,
                                    int batchCount)
    */
    checkCublasStatus(cublasHgemmStridedBatched(
        handle,
        OP_A,
        OP_B,
        m,
        n,
        k,
        (const __half *)&alpha,
        (const __half *)a.const_data_ptr<at::Half>(),
        lda,
        stride_a,
        (const __half *)b.const_data_ptr<at::Half>(),
        ldb,
        stride_b,
        (const __half *)&beta,
        (__half *)out.mutable_data_ptr<at::Half>(),
        ldc,
        stride_c,
        int(batch_sz)));
    return out;
}

torch::Tensor cublas_hgemm_batched_impl_simple(torch::Tensor a, torch::Tensor b)
{
    auto A = b;
    auto B = a;

    int M = A.size(0);
    int N = B.size(1);
    int K = A.size(1);

    int lda = A.size(1);
    int ldb = B.size(2);
    int ldc = M;
    int out_h = N;
    int out_w = M;

    bool trans_a = true;
    bool trans_b = false;

    return cublas_hgemm_batched_kernel(A, B, M, N, K, trans_a, trans_b, lda, ldb, ldc, out_h, out_w, true);
}

torch::Tensor cublas_hgemm_batched_impl_custom(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int out_h,
    int out_w,
    bool trans_a,
    bool trans_b,
    bool weight_is_a)
{
    return cublas_hgemm_batched_kernel(a, b, m, n, k, trans_a, trans_b, lda, ldb, ldc, out_h, out_w, weight_is_a);
}
