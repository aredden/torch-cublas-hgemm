
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Handle.h>

void CUDNN_CHECK(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("CUDNN error: %d, %s\n", status, cudnnGetErrorString(status));
        throw std::runtime_error(cudnnGetErrorString(status));
    }
}

inline void CUDA_CHECK(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

torch::Tensor
cudnn_conv2d_kernel(torch::Tensor input, torch::Tensor weight, torch::Tensor bias = {}, int64_t stride = 1, int64_t padding = 1, torch::Tensor workspace = {}, int64_t groups = 1, int64_t dilation = 1) {
    // Get the dimensions of the input and weight tensors
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t input_height = input.size(2);
    int64_t input_width = input.size(3);
    int64_t output_channels = weight.size(0);
    int64_t kernel_height = weight.size(2);
    int64_t kernel_width = weight.size(3);

    // Compute the output dimensions
    int64_t output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    int64_t output_width = (input_width + 2 * padding - kernel_width) / stride + 1;

    // Create the output tensor
    auto output = torch::empty({batch_size, output_channels, output_height, output_width}, input.options().device(input.device()));

    // Create cuDNN handles and descriptors
    cudnnHandle_t cudnn = at::native::getCudnnHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudnnTensorDescriptor_t input_desc, output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set the tensor descriptors
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batch_size, input_channels, input_height, input_width
    ));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batch_size, output_channels, output_height, output_width
    ));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, output_channels, input_channels, kernel_height, kernel_width
    ));

    // Set the convolution descriptor
    // CUDNN_
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, padding, padding, stride, stride, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF
    ));
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, groups));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    // Perform the convolution
    const half alpha = 1.0f, beta = 0.0f;
    cudaStreamSynchronize(stream);
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_desc,
        input.data_ptr<at::Half>(),
        filter_desc,
        weight.data_ptr<at::Half>(),
        conv_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        workspace.data_ptr<uint8_t>(),
        workspace.numel() * workspace.element_size(),
        &beta,
        output_desc,
        output.data_ptr<at::Half>()
    ));
    cudaStreamSynchronize(stream);

    // Add bias if provided
    if (bias.defined()) {
        CUDNN_CHECK(cudnnAddTensor(
            cudnn, &alpha, output_desc, bias.data_ptr<at::Half>(), &alpha, output_desc, output.data_ptr<at::Half>()
        ));
    }

    // Destroy cuDNN handles and descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    return output;
}
