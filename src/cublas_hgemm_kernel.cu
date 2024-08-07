#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

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

torch::Tensor cublas_gemm_kernel(torch::Tensor a, torch::Tensor b, int m, int n, int k, bool trans_a = false, bool trans_b = false, int lda = -1, int ldb = -1, int ldc = -1, int out_h = -1, int out_w = -1)
{
    c10::cuda::CUDAGuard device_guard(a.device());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    at::Half alpha = 1.0f;
    at::Half beta = 0.0f;

    if (out_h == -1)
    {
        out_h = m;
    }

    if (out_w == -1)
    {
        out_w = n;
    }

    torch::Tensor out = torch::empty({out_h, out_w}, a.options().device(a.device()));

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
    cublasStatus_t result = cublasHgemm(
        handle,
        OP_A,
        OP_B,
        m,
        n,
        k,
        (__half *)&alpha,
        (__half *)a.const_data_ptr<at::Half>(),
        lda,
        (__half *)b.const_data_ptr<at::Half>(),
        ldb,
        (__half *)&beta,
        (__half *)out.mutable_data_ptr<at::Half>(),
        ldc);
    if (result != CUBLAS_STATUS_SUCCESS)
    {
        const char *results = cublasGetStatusString(result);

        std::string error_message = "cublasGemmEx failed with error: ";
        error_message += results;

        std::cout << error_message << std::endl;

        throw std::runtime_error(results);
    }

    return out;
}

// b is equal to A, and a is equal to B, because cublas is opposite land.
torch::Tensor cublas_gemm_kernel_axbT(torch::Tensor a, torch::Tensor b)
{
    at::Half alpha = 1.0f;
    at::Half beta = 0.0f;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);

    int lda = a.size(1);
    int ldb = b.size(1);

    int ldc = m;
    int out_h = n;
    int out_w = m;

    // output as (N, M) - because opposite land.
    torch::Tensor out = torch::empty({out_h, out_w}, a.options().device(a.device()));

    /*
    ### REFERENCE ###
    cublasStatus_t cublasHgemm(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        const half *alpha,
        const half *A,
        int lda,
        const half *B,
        int ldb,
        const half *beta,
        half *C,
        int ldc
    );
    */
    checkCublasStatus(cublasHgemm(
        handle,
        // Transpose a
        CUBLAS_OP_T,
        // Don't transpose b
        CUBLAS_OP_N,
        m,
        n,
        k,
        (__half *)&alpha,
        (__half *)a.const_data_ptr<at::Half>(),
        lda,
        (__half *)b.const_data_ptr<at::Half>(),
        ldb,
        (__half *)&beta,
        (__half *)out.mutable_data_ptr<at::Half>(),
        ldc));

    return out;
}