from typing import Literal, Optional

import torch

from cublas_ops_ext import cublas_hgemm_axbT as _cublas_hgemm_axbT
from cublas_ops_ext import cublas_hgemm_batched_simple as _cublas_hgemm_batched_simple
from cublas_ops_ext import cublaslt_hgemm_simple as _cublaslt_hgemm_simple
from cublas_ops_ext import cublaslt_hgemm_batched_simple as _cublaslt_hgemm_batched_simple
from torch import nn

global has_moved
has_moved = {
    idx: False for idx in range(torch.cuda.device_count())
}

class StaticState:
    workspace = {idx:torch.empty((1024 * 1024 * 8,), dtype=torch.uint8) for idx in range(torch.cuda.device_count())}
    workspace_size = workspace[0].nelement()
    bias_g = {idx:torch.tensor([], dtype=torch.float16) for idx in range(torch.cuda.device_count())}

    @classmethod
    def get(cls, __name: str, device: torch.device) -> torch.Any:
        global has_moved
        idx = device.index if device.index is not None else 0
        if not has_moved[idx]:
            cls.workspace = cls.workspace[idx].cuda(idx)
            cls.bias_g = cls.bias_g[idx].cuda(idx)
            has_moved[idx] = True
        if "bias" in __name:
            return cls.bias_g[idx]
        if "workspace" in __name:
            return cls.workspace[idx]
        if "workspace_size" in __name:
            return cls.workspace_size


@torch.inference_mode()
def cublas_half_matmul_batched_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_batched_simple(a, b)


@torch.inference_mode()
def cublas_half_matmul_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_axbT(b, a)


@torch.inference_mode
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


@torch.inference_mode
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
        **state_kwargs,
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self._epilogue_str = epilogue_str
        self.bias_ref = None if not bias else self.bias
        self.has_bias = bias

    def forward(self, x):
        if x.dtype != torch.float16 or self.weight.device.type != "cuda":
            out = super().forward(x)
            if self._epilogue_str == "RELU":
                out = torch.relu(out)
            elif self._epilogue_str == "GELU":
                out = torch.nn.functional.gelu(out)
            return out

        use_cublasLt = (self.has_bias or self._epilogue_str != "NONE")
        if x.ndim == 1:
            x = x.unsqueeze(0)
        dim2or3 = x.ndim in [2, 3]
        if dim2or3 and not use_cublasLt:
            if x.ndim == 3:
                return cublas_half_matmul_batched_simple(x.contiguous(), self.weight)
            else:
                return cublas_half_matmul_simple(x.contiguous(), self.weight)
        elif dim2or3 and use_cublasLt:
            if x.ndim == 3:
                return cublaslt_fused_half_matmul_batched_simple(
                    x.contiguous(), self.weight, bias=self.bias_ref, epilogue_str=self._epilogue_str
                )
            else:
                return cublaslt_fused_half_matmul_simple(
                    x.contiguous(), self.weight, bias=self.bias_ref, epilogue_str=self._epilogue_str
                )
        else:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        out = cublaslt_fused_half_matmul_simple(
            x.contiguous(), self.weight, bias=self.bias_ref, epilogue_str=self._epilogue_str
        )
        return out.view(*leading_dims, out.shape[-1])


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
    "cublaslt_fused_half_matmul_batched_simple"
]