"""
SageMaker L40S (sm_89) NVRTC workaround.

PyTorch 2.6 has no precompiled reduction kernels for integer types on sm_89.
It falls back to NVRTC JIT compilation, which fails because SageMaker injects
libnvrtc.so.13.0 (driver-bundled) but not libnvrtc-builtins.so.13.0 (toolkit-bundled).

Fix: monkey-patch integer tensor reductions to run on CPU. These are tiny ops
(grid sizes, shape calculations) so CPU is fine.
"""
import torch

_orig_prod = torch.Tensor.prod
_orig_sum = torch.Tensor.sum
_orig_cumsum = torch.Tensor.cumsum
_orig_cumprod = torch.Tensor.cumprod
_orig_torch_prod = torch.prod
_orig_torch_sum = torch.sum
_orig_torch_cumsum = torch.cumsum
_orig_torch_cumprod = torch.cumprod


def _needs_cpu(t):
    return isinstance(t, torch.Tensor) and t.is_cuda and not t.is_floating_point() and not t.is_complex()


def _safe_prod(self, *args, **kwargs):
    if _needs_cpu(self):
        kwargs.pop("out", None)
        result = _orig_prod(self.cpu(), *args, **kwargs)
        return result.to(self.device) if isinstance(result, torch.Tensor) else result
    return _orig_prod(self, *args, **kwargs)


def _safe_sum(self, *args, **kwargs):
    if _needs_cpu(self):
        kwargs.pop("out", None)
        result = _orig_sum(self.cpu(), *args, **kwargs)
        return result.to(self.device) if isinstance(result, torch.Tensor) else result
    return _orig_sum(self, *args, **kwargs)


def _safe_cumsum(self, *args, **kwargs):
    if _needs_cpu(self):
        kwargs.pop("out", None)
        return _orig_cumsum(self.cpu(), *args, **kwargs).to(self.device)
    return _orig_cumsum(self, *args, **kwargs)


def _safe_cumprod(self, *args, **kwargs):
    if _needs_cpu(self):
        kwargs.pop("out", None)
        return _orig_cumprod(self.cpu(), *args, **kwargs).to(self.device)
    return _orig_cumprod(self, *args, **kwargs)


def _safe_torch_prod(input, *args, **kwargs):
    if _needs_cpu(input):
        kwargs.pop("out", None)
        result = _orig_torch_prod(input.cpu(), *args, **kwargs)
        return result.to(input.device) if isinstance(result, torch.Tensor) else result
    return _orig_torch_prod(input, *args, **kwargs)


def _safe_torch_sum(input, *args, **kwargs):
    if _needs_cpu(input):
        kwargs.pop("out", None)
        result = _orig_torch_sum(input.cpu(), *args, **kwargs)
        return result.to(input.device) if isinstance(result, torch.Tensor) else result
    return _orig_torch_sum(input, *args, **kwargs)


def _safe_torch_cumsum(input, *args, **kwargs):
    if _needs_cpu(input):
        kwargs.pop("out", None)
        return _orig_torch_cumsum(input.cpu(), *args, **kwargs).to(input.device)
    return _orig_torch_cumsum(input, *args, **kwargs)


def _safe_torch_cumprod(input, *args, **kwargs):
    if _needs_cpu(input):
        kwargs.pop("out", None)
        return _orig_torch_cumprod(input.cpu(), *args, **kwargs).to(input.device)
    return _orig_torch_cumprod(input, *args, **kwargs)


torch.Tensor.prod = _safe_prod
torch.Tensor.sum = _safe_sum
torch.Tensor.cumsum = _safe_cumsum
torch.Tensor.cumprod = _safe_cumprod
torch.prod = _safe_torch_prod
torch.sum = _safe_torch_sum
torch.cumsum = _safe_torch_cumsum
torch.cumprod = _safe_torch_cumprod
