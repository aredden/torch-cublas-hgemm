#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define DEFAULT_WORKSPACE_SIZE 134217728

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

cublasLtMatrixLayout_t createMatrixLayout(
    torch::Tensor tensor, uint64_t rows, uint64_t cols, int64_t ld, int64_t stride = 0, int batch_sz = 1)
{
    cublasLtMatrixLayout_t layout = nullptr;

    // std::cout << "Creating matrix layout" << std::endl;
    checkCublasStatus(cublasLtMatrixLayoutCreate(&layout, CUDA_R_16F, rows, cols, ld));

    // std::cout << "Setting batch count" << std::endl;
    checkCublasStatus(
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_sz, sizeof(batch_sz)));
    // std::cout << "Setting stride" << std::endl;
    checkCublasStatus(
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));

    return layout;
}

cublasLtEpilogue_t parseEpilogue(std::string epilogue_str, bool has_bias)
{
    cublasLtEpilogue_t epilogue;

    if (epilogue_str == "NONE")
    {
        epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    }
    else if (epilogue_str == "RELU")
    {
        epilogue = has_bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
    }
    else if (epilogue_str == "GELU")
    {
        epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
    }
    else
    {
        throw std::invalid_argument("Invalid epilogue type");
    }

    return epilogue;
}

torch::Tensor cublaslt_gemm_batched_launch_axbT(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias,
    std::string epilogue_str,
    torch::Tensor workspace,
    size_t workspaceSize,
    bool trans_a,
    bool trans_b,
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    int out_h,
    int out_w)
{
    c10::cuda::CUDAGuard device_guard(a.device());

    cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

    bool has_bias = bias.numel() > 0;

    if (workspace.numel() == 0)
    {
        // Allocate workspace if not provided
        workspace = at::empty(workspaceSize, at::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    }

    int32_t a_batch_stride = a.ndimension() == 3 ? a.stride(0) : 0;
    int32_t b_batch_stride = b.ndimension() == 3 ? b.stride(0) : 0;

    int32_t batch_sz;
    if (a.ndimension() == 3 && b.ndimension() == 3)
    {
        batch_sz = max(a.size(0), b.size(0));
    }
    else if (a.ndimension() == 3)
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

    // Get bias pointer and data type
    // Will only be used if has_bias is true
    half *bias_ref = (half *)bias.const_data_ptr<at::Half>();
    auto cublasBiasDataType = CUDA_R_16F;

    // Allocate output tensor
    torch::Tensor out = at::empty({batch_sz, out_h, out_w}, a.options().device(a.device()));

    int32_t out_stride = out.stride(0);

    // Create cublasLtMatmulDesc_t's, cublasLtMatrixLayout_t's, and cublasLtMatmulPreference_t
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t aDesc = nullptr, bDesc = nullptr, outDesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    // Handle transposition
    cublasOperation_t transa = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Create matmul operation descriptor with fp16 accumulation
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));

    // Set transposition attributes
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // std::cout << "Parsing epilogue" << std::endl;

    // Parse epilogue type
    cublasLtEpilogue_t epilogue = parseEpilogue(epilogue_str, has_bias);

    // std::cout << "Setting epilogue" << std::endl;

    // Set epilogue attribute in operation descriptor
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    // std::cout << "Set epilogue, adding bias" << std::endl;

    // Set bias attributes if bias is provided
    if (has_bias)
    {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ref, sizeof(bias_ref)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &cublasBiasDataType, sizeof(cublasBiasDataType)));
        // checkCublasStatus(
        //     cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, &bias_stride, sizeof(bias_stride))
        // );
    }

    // Create matrix layout descriptors based on input tensors shapes

    // std::cout << "Creating matrix layout A" << std::endl;
    aDesc = createMatrixLayout(
        a, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda, a_batch_stride, batch_sz);
    // std::cout << "Creating matrix layout B" << std::endl;
    bDesc = createMatrixLayout(
        b, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb, b_batch_stride, batch_sz);
    // std::cout << "Creating matrix layout out" << std::endl;
    outDesc = createMatrixLayout(out, m, n, ldc, out_stride, batch_sz);

    // Create preference descriptor & set workspace size
    // std::cout << "Creating preference" << std::endl;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));

    // std::cout << "Setting preference" << std::endl;
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    /*


### FOR REFERENCE ###

cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t computeDesc,
    const void *alpha, const void *A,
    cublasLtMatrixLayout_t Adesc,
    const void *B,
    cublasLtMatrixLayout_t Bdesc,
    const void *beta,
    const void *C,
    cublasLtMatrixLayout_t Cdesc,
    void *D,
    cublasLtMatrixLayout_t Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void *workspace,
    size_t workspaceSizeInBytes,
    cudaStream_t stream
)
*/
    const at::Half alpha = 1.0f;
    const at::Half beta = 0.0f;
    // CUDA GO NYOOM NYOOM
    checkCublasStatus(cublasLtMatmul(
        ltHandle,
        operationDesc,
        (const void *)&alpha,
        (const void *)a.const_data_ptr<at::Half>(),
        aDesc,
        (const void *)b.const_data_ptr<at::Half>(),
        bDesc,
        (const void *)&beta,
        (const void *)out.mutable_data_ptr<at::Half>(),
        outDesc,
        (void *)out.mutable_data_ptr<at::Half>(),
        outDesc,
        nullptr,
        (void *)workspace.mutable_data_ptr<uint8_t>(),
        workspaceSize,
        at::cuda::getCurrentCUDAStream(a.device().index())));

    // Clean up
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(outDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return out;
}

torch::Tensor cublaslt_hgemm_batched_impl_simple(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {})
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

    return cublaslt_gemm_batched_launch_axbT(
        A,
        B,
        bias,
        epilogue_str,
        workspace,
        workspace.numel() * workspace.element_size(),
        trans_a,
        trans_b,
        M,
        N,
        K,
        lda,
        ldb,
        ldc,
        out_h,
        out_w);
}