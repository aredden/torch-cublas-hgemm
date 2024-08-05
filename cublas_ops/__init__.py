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
def hgemv_simt(vec: torch.HalfTensor, mat: torch.HalfTensor, block_dim_x: int = 32):
    prev_dims = vec.shape[:-1]
    return _simt_hgemv(mat, vec.view(-1, 1), block_dim_x=block_dim_x).view(
        *prev_dims, -1
    )


@torch.no_grad()
def cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_batched_simple(a, b)


@torch.no_grad()
def cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_axbT(b, a)


@torch.no_grad()
def cublaslt_fused_half_matmul_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


@torch.no_grad()
def cublaslt_fused_half_matmul_batched_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias", a.device)
    return _cublaslt_hgemm_batched_simple(
        a, b, bias, epilogue_str, StaticState.get("workspace", a.device)
    )


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

    def forward(self, x):
        if x.dtype != torch.float16 or self.weight.device.type != "cuda":
            out = super().forward(x)
            if self._epilogue_str == "RELU":
                out = torch.relu(out)
            elif self._epilogue_str == "GELU":
                out = torch.nn.functional.gelu(out)
            return out

        use_cublasLt = self.has_bias or self._epilogue_str != "NONE"
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if math.prod(x.shape) == x.shape[-1]:
            out = hgemv_simt(x, self.weight)
            if not use_cublasLt:
                return out
            else:
                if self.has_bias:
                    out += self.bias.data
                if self._epilogue_str == "RELU":
                    return torch.relu(out)
                elif self._epilogue_str == "GELU":
                    return torch.nn.functional.gelu(out)
                return out
        if not use_cublasLt:
            if x.ndim == 3:
                return cublas_half_matmul_batched_simple(x, self.weight)
            elif x.ndim == 2:
                return cublas_half_matmul_simple(x, self.weight)
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublas_half_matmul_simple(x, self.weight).view(
                *leading_dims, out.shape[-1]
            )
        if use_cublasLt:
            if x.ndim == 3:
                return cublaslt_fused_half_matmul_batched_simple(
                    x, self.weight, bias=self.bias.data, epilogue_str=self._epilogue_str
                )
            elif x.ndim == 2:
                return cublaslt_fused_half_matmul_simple(
                    x, self.weight, bias=self.bias.data, epilogue_str=self._epilogue_str
                )

            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
            out = cublaslt_fused_half_matmul_simple(
                x, self.weight, bias=self.bias.data, epilogue_str=self._epilogue_str
            ).view(*leading_dims, out.shape[-1])
        return out


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
