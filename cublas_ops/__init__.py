from typing import Literal, Optional

import torch
from cublas_ops_ext import cublas_hgemm_axbT as _cublas_hgemm_axbT
from cublas_ops_ext import cublas_hgemm_batched_simple as _cublas_hgemm_batched_simple
from cublas_ops_ext import cublaslt_hgemm_simple as _cublaslt_hgemm_simple
from torch import nn

has_moved = False
class StaticState:
    workspace = torch.empty((1024 * 1024 * 8,), dtype=torch.uint8)
    workspace_size = workspace.nelement()
    bias_g = torch.tensor([], dtype=torch.float16)

    @classmethod
    def get(cls, __name: str) -> torch.Any:
        global has_moved
        if not has_moved:
            cls.workspace = cls.workspace.cuda()
            cls.bias_g = cls.bias_g.cuda()
            has_moved = True
        if "bias" in __name:
            return cls.bias_g
        if "workspace" in __name:
            return cls.workspace
        if "workspace_size" in __name:
            return cls.workspace_size


@torch.inference_mode()
def cublaslt_matmul_batched_simple_axbT(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_batched_simple(a, b)


@torch.inference_mode()
def cublas_hgemm_simple(a: torch.Tensor, b: torch.Tensor):
    return _cublas_hgemm_axbT(b, a)


@torch.inference_mode
def cublaslt_hgemm_simple(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    epilogue_str: Optional[Literal["NONE", "RELU", "GELU"]] = "NONE",
):
    if bias is None:
        bias = StaticState.get("bias")
    return _cublaslt_hgemm_simple(a, b, bias, epilogue_str, StaticState.get("workspace"))


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
        self.extra_kwargs = {"epilogue_str": epilogue_str, **state_kwargs}
        self._epilogue_str = epilogue_str
        self.bias_ref = StaticState.get('bias') if not bias else self.bias
        self.has_bias = bias


    def forward(self, x):
        if x.dtype != torch.float16 or self.weight.device.type != "cuda":
            out = super().forward(x)
            if self._epilogue_str == "RELU":
                out = torch.relu(out)
            elif self._epilogue_str == "GELU":
                out = torch.nn.functional.gelu(out)
            return out

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim in [2, 3] and not self.has_bias and self._epilogue_str == "NONE":
            if x.ndim == 3:
                return cublaslt_matmul_batched_simple_axbT(
                    x.contiguous(), self.weight
                )
            else:
                return cublas_hgemm_simple(x.contiguous(), self.weight)
        else:
            leading_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        out = cublaslt_hgemm_simple(
            x.contiguous(), self.weight, bias=self.bias_ref, **self.extra_kwargs
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
