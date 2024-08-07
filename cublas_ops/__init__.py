import math
from typing import Literal, Optional

import torch
from cublas_ops_ext import _simt_hgemv
from cublas_ops_ext import cublas_hgemm_axbT as _cublas_hgemm_axbT
from cublas_ops_ext import cublas_hgemm_batched_simple as _cublas_hgemm_batched_simple
from cublas_ops_ext import (
    cublaslt_hgemm_batched_simple as _cublaslt_hgemm_batched_simple,
)
from cublas_ops_ext import cublaslt_hgemm_simple as _cublaslt_hgemm_simple
from torch import nn
from torch.nn import functional as F

global has_moved
has_moved = {idx: False for idx in range(torch.cuda.device_count())}


class StaticState:
    workspace = {
        idx: torch.empty((1024 * 1024 * 8,), dtype=torch.uint8)
        for idx in range(torch.cuda.device_count())
    }
    workspace_size = workspace[0].nelement()
    bias_g = {
        idx: torch.tensor([], dtype=torch.float16)
        for idx in range(torch.cuda.device_count())
    }

    @classmethod
    def get(cls, __name: str, device: torch.device) -> torch.Any:
        global has_moved
        idx = device.index if device.index is not None else 0
        if not has_moved[idx]:
            cls.workspace[idx] = cls.workspace[idx].cuda(idx)
            cls.bias_g[idx] = cls.bias_g[idx].cuda(idx)
            has_moved[idx] = True
        if "bias" in __name:
            return cls.bias_g[idx]
        if "workspace" in __name:
            return cls.workspace[idx]
        if "workspace_size" in __name:
            return cls.workspace_size


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::hgemv_simt", device_types=["cuda"], mutates_args=()
)
def hgemv_simt(
    vec: torch.Tensor, mat: torch.Tensor, block_dim_x: int = 32
) -> torch.Tensor:
    prev_dims = vec.shape[:-1]
    return _simt_hgemv(mat, vec.view(-1, 1), block_dim_x=block_dim_x).view(
        *prev_dims, -1
    )


@torch.no_grad()
@torch.library.register_fake("cublas_ops_ext::hgemv_simt")
def _(vec: torch.Tensor, mat: torch.Tensor, block_dim_x: int = 32) -> torch.Tensor:
    # prev_dims = vec.shape[:-1]
    return F.linear(vec.view(1,-1), mat, None)
    # return (mat @ vec.reshape(-1, 1).contiguous()).view(*prev_dims, -1)


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::cublas_hgemm_batched_simple",
    device_types=["cuda"],
    mutates_args=(),
)
def cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _cublas_hgemm_batched_simple(a, b)


@torch.library.register_fake("cublas_ops_ext::cublas_hgemm_batched_simple")
def _(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.linear(a, b, None)


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::cublas_hgemm_simple", device_types=["cuda"], mutates_args=()
)
def cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _cublas_hgemm_axbT(b, a)


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::cublaslt_fused_half_matmul_simple",
    device_types=["cuda"],
    mutates_args=(),
)
def cublaslt_fused_half_matmul_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[str] = None,
) -> torch.Tensor:
    if epilogue_str is None:
        epilogue_str = "NONE"
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::cublaslt_fused_half_matmul_batched_simple",
    device_types=["cuda"],
    mutates_args=(),
)
def cublaslt_fused_half_matmul_batched_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[str] = None,
) -> torch.Tensor:
    if epilogue_str is None:
        epilogue_str = "NONE"
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_batched_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


@torch.no_grad()
@torch.library.custom_op(
    "cublas_ops_ext::cublas_half_matmul",
    device_types=["cuda"],
    mutates_args=(),
)
def cublas_half_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[str] = None,
    has_bias: bool = False,
) -> torch.Tensor:
    if epilogue_str is None:
        epilogue_str = "NONE"
    out_dtype = x.dtype
    needs_convert = out_dtype != torch.float16
    if needs_convert:
        x = x.type(torch.float16)
    use_cublasLt = has_bias or epilogue_str != "NONE"
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if math.prod(x.shape) == x.shape[-1]:
        out = hgemv_simt(x, weight)
        if use_cublasLt:
            if has_bias:
                out = out + bias.broadcast_to(out.shape)
            if epilogue_str == "RELU":
                out = F.relu(out)
            elif epilogue_str == "GELU":
                out = F.gelu(out)
        if needs_convert:
            out_final = out.type(out_dtype)
        else:
            out_final = out
    elif not use_cublasLt:
        if x.ndim == 3:
            out = cublas_half_matmul_batched_simple(x, weight)
        elif x.ndim == 2:
            out = cublas_half_matmul_simple(x, weight)
        else:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublas_half_matmul_simple(x, weight).view(
                *leading_dims, out.shape[-1]
            )
        if needs_convert:
            out_final = out.type(out_dtype)
        else:
            out_final = out
    else:
        if x.ndim == 3:
            out = cublaslt_fused_half_matmul_batched_simple(
                x, weight, bias=bias.data, epilogue_str=epilogue_str
            )
        elif x.ndim == 2:
            out = cublaslt_fused_half_matmul_simple(
                x, weight, bias=bias.data, epilogue_str=epilogue_str
            )
        else:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublaslt_fused_half_matmul_simple(
                x, weight, bias=bias.data, epilogue_str=epilogue_str
            ).view(*leading_dims, out.shape[-1])
        if needs_convert:
            out_final = out.type(out_dtype)
        else:
            out_final = out
    return out_final



@torch.no_grad()
@torch.library.register_fake(
    "cublas_ops_ext::cublas_half_matmul",
)
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[str] = None,
    has_bias: bool = False,
) -> torch.Tensor:
    if epilogue_str is None:
        epilogue_str = "NONE"

    out_dtype = x.dtype
    needs_convert = out_dtype != torch.float16

    if needs_convert:
        x = x.type(torch.float16)
    return F.linear(x, weight, bias)


class CublasLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
        epilogue_str="NONE",
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._epilogue_str = epilogue_str
        self.has_bias = bias
        self.has_checked_weight = False

    def forward(self, x):
        if not self.has_checked_weight:
            if not self.weight.dtype == torch.float16:
                self.to(dtype=torch.float16)
            self.has_checked_weight = True
        return cublas_half_matmul(
            x, self.weight, self.bias, self._epilogue_str, self.has_bias
        )


class CublasLinearGelu(CublasLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            epilogue_str="GELU",
        )


class CublasLinearRelu(CublasLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=torch.float16,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            epilogue_str="RELU",
        )


__ALL__ = [
    "CublasLinear",
    "CublasLinearGelu",
    "CublasLinearRelu",
    "cublas_half_matmul_simple",
    "cublas_half_matmul_batched_simple",
    "cublaslt_fused_half_matmul_simple",
    "cublaslt_fused_half_matmul_batched_simple",
]
