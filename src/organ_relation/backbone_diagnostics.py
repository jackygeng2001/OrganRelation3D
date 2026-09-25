"""Diagnostic hooks; no training policy or automatic device selection."""
from __future__ import annotations
import torch


def parameter_inventory(module):
    named=list(module.named_parameters())
    return {
        'parameter_count':sum(p.numel() for _,p in named),
        'trainable_count':sum(p.numel() for _,p in named if p.requires_grad),
        'parameter_bytes':sum(p.numel()*p.element_size() for _,p in named),
        'gradient_bytes':sum(p.grad.numel()*p.grad.element_size() for _,p in named if p.grad is not None),
        'missing_gradients':[name for name,p in named if p.requires_grad and p.grad is None],
        'nonfinite_gradients':[name for name,p in named if p.grad is not None and not bool(torch.isfinite(p.grad).all())],
        'zero_gradient_tensors':[name for name,p in named if p.grad is not None and not bool(p.grad.abs().sum()>0)],
    }


class DeviceMemoryMonitor:
    """Cumulative allocator peaks since begin(), for caller-selected device.

    PyTorch ROCm uses the torch.cuda allocator API too. CPU fields are None,
    never zero pretending to be measured GPU memory. Does not reserve/free
    memory, empty caches or change model tensors. GPU use must be explicitly
    selected by the entry point; no fallback to another device.
    """
    def __init__(self,device):
        self.device=torch.device(device)
        if self.device.type not in ('cpu','cuda'):
            raise ValueError('only CPU or PyTorch CUDA/ROCm devices supported')
        self.started=False

    def begin(self):
        if self.device.type=='cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('requested GPU backend is unavailable; no CPU fallback')
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started=True

    def snapshot(self,phase):
        if not self.started:
            raise RuntimeError('call begin before taking a memory snapshot')
        info={'phase':phase,'device':str(self.device),'backend':'cpu',
              'allocated_bytes':None,'reserved_bytes':None,
              'peak_allocated_bytes':None,'peak_reserved_bytes':None}
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            info.update(backend='rocm' if torch.version.hip else 'cuda',
                        allocated_bytes=torch.cuda.memory_allocated(self.device),
                        reserved_bytes=torch.cuda.memory_reserved(self.device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return info
