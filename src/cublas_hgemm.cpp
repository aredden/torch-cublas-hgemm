#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define DEFAULT_WORKSPACE_SIZE 134217728
void initGemvCustomBindings(py::module &m);
torch::Tensor cublaslt_hgemm_batched_impl_simple(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {}
);
torch::Tensor cublas_hgemm_batched_impl_simple(torch::Tensor a, torch::Tensor b);
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
    bool weight_is_a
);
torch::Tensor cublas_gemm_kernel_axbT(torch::Tensor a, torch::Tensor b);
torch::Tensor cublas_gemm_kernel(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    bool trans_a = false,
    bool trans_b = false,
    int lda = -1,
    int ldb = -1,
    int ldc = -1,
    int out_h = -1,
    int out_w = -1
);
torch::Tensor cublaslt_hgemm_simple(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor bias = {},
    std::string epilogue_str = "NONE",
    torch::Tensor workspace = {}
);
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
);
torch::Tensor cublas_hgemm(
    torch::Tensor a,
    torch::Tensor b,
    int m,
    int n,
    int k,
    bool trans_a = false,
    bool trans_b = false,
    int lda = -1,
    int ldb = -1,
    int ldc = -1,
    int out_h = -1,
    int out_w = -1
) {
    return cublas_gemm_kernel(a, b, m, n, k, trans_a, trans_b, lda, ldb, ldc, out_h, out_w);
}
torch::Tensor cublas_hgemm_axbT(torch::Tensor a, torch::Tensor b) {
    return cublas_gemm_kernel_axbT(a, b);
}

torch::Tensor cublasLt_hgemm_custom_impl(
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
    size_t workspaceSize = 134217728UL
) {
    return cublaslt_hgemm_customizable(
        a, b, bias, epilogue_str, workspace, trans_a, trans_b, m, n, k, lda, ldb, ldc, out_h, out_w, workspaceSize
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cublas_hgemm",
        py::overload_cast<torch::Tensor, torch::Tensor, int, int, int, bool, bool, int, int, int, int, int>(
            &cublas_hgemm
        ),
        py::arg("a"),
        py::arg("b"),
        py::arg("m"),
        py::arg("n"),
        py::arg("k"),
        py::arg("trans_a") = false,
        py::arg("trans_b") = false,
        py::arg("lda") = -1,
        py::arg("ldb") = -1,
        py::arg("ldc") = -1,
        py::arg("out_h") = -1,
        py::arg("out_w") = -1
    );
    m.def(
        "cublas_hgemm_axbT",
        py::overload_cast<torch::Tensor, torch::Tensor>(&cublas_hgemm_axbT),
        py::arg("a"),
        py::arg("b")
    );
    m.def(
        "cublaslt_hgemm_custom",
        py::overload_cast<
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            std::string,
            torch::Tensor,
            bool,
            bool,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            size_t>(&cublasLt_hgemm_custom_impl),
        py::arg("a"),
        py::arg("b"),
        py::arg("bias") = torch::empty({}),
        py::arg("epilogue_str") = "NONE",
        py::arg("workspace") = torch::empty({}),
        py::arg("trans_a") = false,
        py::arg("trans_b") = false,
        py::arg("m") = -1,
        py::arg("n") = -1,
        py::arg("k") = -1,
        py::arg("lda") = -1,
        py::arg("ldb") = -1,
        py::arg("ldc") = -1,
        py::arg("out_h") = -1,
        py::arg("out_w") = -1,
        py::arg("workspaceSize") = 134217728UL
    );
    m.def(
        "cublaslt_hgemm_simple",
        py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, std::string, torch::Tensor>(
            &cublaslt_hgemm_simple
        ),
        py::arg("a"),
        py::arg("b"),
        py::arg("bias") = torch::empty({}),
        py::arg("epilogue_str") = "NONE",
        py::arg("workspace") = torch::empty({})
    );

    m.def(
        "cublaslt_hgemm_batched_simple",
        py::overload_cast<torch::Tensor, torch::Tensor, torch::Tensor, std::string, torch::Tensor>(
            &cublaslt_hgemm_batched_impl_simple
        ),
        py::arg("a"),
        py::arg("b"),
        py::arg("bias") = torch::empty({}),
        py::arg("epilogue_str") = "NONE",
        py::arg("workspace") = torch::empty({})
    );

    m.def(
        "cublas_hgemm_batched_simple",
        py::overload_cast<torch::Tensor, torch::Tensor>(&cublas_hgemm_batched_impl_simple),
        py::arg("a"),
        py::arg("b")
    );
    m.def(
        "cublas_hgemm_batched_custom",
        py::overload_cast<torch::Tensor, torch::Tensor, int, int, int, int, int, int, int, int, bool, bool, bool>(
            &cublas_hgemm_batched_impl_custom
        ),
        py::arg("a"),
        py::arg("b"),
        py::arg("m") = -1,
        py::arg("n") = -1,
        py::arg("k") = -1,
        py::arg("lda") = -1,
        py::arg("ldb") = -1,
        py::arg("ldc") = -1,
        py::arg("out_h") = -1,
        py::arg("out_w") = -1,
        py::arg("trans_a") = true,
        py::arg("trans_b") = false,
        py::arg("weight_is_a") = true,
        "cublas hgemm batched custom implementation"
    );
    initGemvCustomBindings(m);
}