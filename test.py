import torch
from torch.utils.benchmark import Timer
from torch import nn
from cublas_ops import CublasLinear


def time_fn(fn, A, B, bias, warmups=20, iters=50):
    def fn_():
        return fn(A, B, bias)

    for _ in range(warmups):
        fn_()

    timer = Timer(stmt="fn_()", globals=locals())
    return timer.timeit(iters)


def time_linear(linear, input, warmups=20, iters=50):
    def fn_():
        return linear(input)

    for _ in range(warmups):
        fn_()

    timer = Timer(stmt="fn_()", globals=locals())
    return timer.timeit(iters)

with torch.no_grad():

    M = 2048*2
    K = 2048*2
    N = 2048*2
    base_module = nn.Linear(in_features=K, out_features=N, bias=True).cuda(0).half()

    # f16_module = F16Linear(in_features=K, out_features=N, bias=True).cuda(0).half()
    cublas_module = (
        CublasLinear(in_features=K, out_features=N, bias=True).cuda(0).half()
    )
    cublas_module.weight.data = base_module.weight.data.clone()
    cublas_module.bias.data = base_module.bias.data.clone()

    cublas_module
    base_module.compile()
    # f16_module.weight.data = base_module.weight.data.clone()
    # f16_module.bias.data = base_module.bias.data.clone()

    input_t = torch.randn(M, K).half().cuda(0) / 4
    
    out = base_module(input_t)
    # out_f16 = f16_module(input_t.clone())
    out_cublas = cublas_module(input_t)

    print("Output From nn.Linear (compiled):\n",out)
    print("Output From CublasLinear:\n",out_cublas)

    FLOPS = 2 * M * N * K

    time_lin = time_linear(base_module, input_t, warmups=50, iters=100)
    time_cublas = time_linear(cublas_module, input_t, warmups=50, iters=100)

    print(
        "torch f16 W/ f32 acc (compiled): ".upper().replace(" ", "_"),
        f"{time_lin.mean * 1000 * 1000:.2f} us",
        f"{(FLOPS / time_lin.mean)/1e12:.2f} TFLOPS",
    )
    print(
        "cublas f16 W/ f16 acc: ".upper().replace(" ", "_"),
        f"{time_cublas.mean * 1000 * 1000:.2f} us",
        f"{(FLOPS / time_cublas.mean)/1e12:.2f} TFLOPS",
    )
