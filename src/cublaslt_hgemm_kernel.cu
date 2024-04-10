#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define DEFAULT_WORKSPACE_SIZE 134217728

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

torch::Tensor cublaslt_gemm_launch_axbT(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {},
    size_t workspaceSize = DEFAULT_WORKSPACE_SIZE,
    bool trans_a = false,
    bool trans_b = false,
    int m = -1,
    int n = -1,
    int k = -1,
    int lda = -1,
    int ldb = -1,
    int ldc = -1,
    int out_h = -1,
    int out_w = -1
) {
    cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(a.device().index());
    bool has_bias = bias.numel() > 0;

    checkCudaStatus(cudaStreamSynchronize(stream));

    if (workspace.numel() == 0) {
        // Allocate workspace if not provided
        workspace = torch::empty(workspaceSize, at::TensorOptions().dtype(torch::kUInt8).device(a.device()));
    }

    // Do confusing stuff with dimensions because magic happens.

    m = m == -1 ? a.size(0) : m;
    n = n == -1 ? b.size(0) : n;
    k = k == -1 ? a.size(1) : k;
    lda = lda == -1 ? a.size(1) : lda;
    ldb = ldb == -1 ? b.size(1) : ldb;
    ldc = ldc == -1 ? m : ldc;
    out_h = out_h == -1 ? n : out_h;
    out_w = out_w == -1 ? m : out_w;


    // Get bias pointer and data type
    // Will only be used if has_bias is true
    half *bias_ref = (half *)bias.const_data_ptr<at::Half>();
    auto cublasBiasDataType = CUDA_R_16F;

    // Allocate output tensor
    torch::Tensor out = torch::empty({out_h, out_w}, a.options().device(a.device()));

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
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa))
    );
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb))
    );

    // Parse epilogue type
    cublasLtEpilogue_t epilogue;


    // Set epilogue attributes
    if (epilogue_str == "NONE") {

        epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    } else if (epilogue_str == "RELU") {

        epilogue = has_bias ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_RELU;
    } else if (epilogue_str == "GELU") {

        epilogue = has_bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU;
    } else {

        throw std::invalid_argument("Invalid epilogue type");
    }
    // Set epilogue attribute in operation descriptor
    checkCublasStatus(
        cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue))
    );

    // Set bias attributes if bias is provided
    if (has_bias) {
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ref, sizeof(bias_ref)
        ));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &cublasBiasDataType, sizeof(cublasBiasDataType)
        ));
    }

    // Create matrix layout descriptors based on input tensors shapes
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &aDesc, CUDA_R_16F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda
    ));
    checkCublasStatus(cublasLtMatrixLayoutCreate(
        &bDesc, CUDA_R_16F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb
    ));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&outDesc, CUDA_R_16F, m, n, ldc));

    // Create preference descriptor & set workspace size
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)
    ));

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

    checkCudaStatus(cudaStreamSynchronize(stream));

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
        (const void *)out.data_ptr<at::Half>(),
        outDesc,
        (void *)out.data_ptr<at::Half>(),
        outDesc,
        nullptr,
        (void *)workspace.data_ptr<uint8_t>(),
        workspaceSize,
        stream
    ));

    // Sync stream
    checkCudaStatus(cudaStreamSynchronize(stream));

    // Clean up
    checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(outDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(bDesc));
    checkCublasStatus(cublasLtMatrixLayoutDestroy(aDesc));
    checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    checkCublasStatus(cublasLtDestroy(ltHandle));

    return out;
}

torch::Tensor cublaslt_hgemm_customizable(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {},
    bool trans_a = false,
    bool trans_b = false,
    int m = -1,
    int n = -1,
    int k = -1,
    int lda = -1,
    int ldb = -1,
    int ldc = -1,
    int out_h = -1,
    int out_w = -1,
    size_t workspaceSize = DEFAULT_WORKSPACE_SIZE
) {
    return cublaslt_gemm_launch_axbT(
        a, b, bias, epilogue_str, workspace, workspaceSize, trans_a, trans_b, m, n, k, lda, ldb, ldc, out_h, out_w
    );
}

torch::Tensor cublaslt_hgemm_simple(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {}
){
    size_t workspace_sz = DEFAULT_WORKSPACE_SIZE;
    if(workspace.numel() > 0){
        workspace_sz = workspace.numel();
    }
    return cublaslt_gemm_launch_axbT(
        b, a, bias, epilogue_str, workspace, workspace_sz, true, false
    );
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def(
//         "cublaslt_hgemm_custom",
//         py::overload_cast<
//             torch::Tensor,
//             torch::Tensor,
//             torch::Tensor,
//             std::string,
//             torch::Tensor,
//             bool,
//             bool,
//             int,
//             int,
//             int,
//             int,
//             int,
//             int,
//             int,
//             int,
//             size_t>(&cublaslt_hgemm_customizable),
//         py::arg("a"),
//         py::arg("b"),
//         py::arg("bias") = torch::empty({}),
//         py::arg("epilogue_str") = "NONE",
//         py::arg("workspace") = torch::empty({}),
//         py::arg("trans_a") = false,
//         py::arg("trans_b") = false,
//         py::arg("m") = -1,
//         py::arg("n") = -1,
//         py::arg("k") = -1,
//         py::arg("lda") = -1,
//         py::arg("ldb") = -1,
//         py::arg("ldc") = -1,
//         py::arg("out_h") = -1,
//         py::arg("out_w") = -1,
//         py::arg("workspaceSize") = 134217728UL
//     );

//     m.def(
//         "cublaslt_hgemm_simple",
//         py::overload_cast<
//             torch::Tensor,
//             torch::Tensor,
//             torch::Tensor,
//             std::string,
//             torch::Tensor>(&cublaslt_hgemm_simple),
//         py::arg("a"),
//         py::arg("b"),
//         py::arg("bias") = torch::empty({}),
//         py::arg("epilogue_str") = "NONE",
//         py::arg("workspace") = torch::empty({})
//     );
// }