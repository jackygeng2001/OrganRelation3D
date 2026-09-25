"""Opt-in eager forward profiling; hooks and dispatch never retain tensors.

ATen boundaries cannot expose allocations internal to a backend convolution.
An OOM identifies the requesting operator, not proof of a particular workspace.
"""
from __future__ import annotations

import math
import time
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def describe(value):
    """JSON-only metadata: no clone, detach, tensor values or storage references."""
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), stride=list(value.stride()),
                    dtype=str(value.dtype), device=str(value.device),
                    bytes=value.numel() * value.element_size(),
                    contiguous=value.is_contiguous(), requires_grad=value.requires_grad)
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return {key: describe(item) for key, item in zip(value._fields, value)}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


class ForwardMemoryProfile(TorchDispatchMode):
    """Call model normally inside this context; backward stays outside it.

    Reset allocator peaks at each boundary and fold each interval into ALL open
    module scopes. This preserves enclosing peaks despite nested resets. Times
    include profiling overhead and nested entries must not be added together.
    CPU memory is null. Failed allocation sizes exist only in backend error text.
    Uses TorchDispatchMode explicitly; an unsupported PyTorch fails on import,
    rather than silently dropping operator coverage. No allocator cache clearing.
    """

    def __init__(self, model, device, result, save):
        super().__init__()
        self.model, self.device = model, torch.device(device)
        self.result, self.save = result, save
        self.active, self.handles = [], []
        result.update(status='running', entries=[], failure=None,
                      last_successful_module=None, last_successful_operator=None,
                      notes=['Per-entry inclusive peaks, including live earlier activations.',
                             'ATen boundaries; backend internal workspace/copies are not individually visible.',
                             'Synchronization and metadata/report overhead affect timings; no tensors retained.'])

    def _memory(self):
        if self.device.type != 'cuda':
            return dict(allocated_bytes=None, reserved_bytes=None,
                        peak_allocated_bytes=None, peak_reserved_bytes=None)
        return dict(allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))

    def _sample(self, *, failed=False):
        if self.device.type == 'cuda' and not failed:
            torch.cuda.synchronize(self.device)
        memory = self._memory()
        for entry in self.active:
            for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
                if memory[key] is not None:
                    entry[key] = max(entry[key], memory[key])
        if self.device.type == 'cuda' and not failed:
            torch.cuda.reset_peak_memory_stats(self.device)
        return memory

    def _start(self, kind, name, args, kwargs):
        memory = self._sample()
        entry = dict(index=len(self.result['entries']), kind=kind, name=name,
                     parent=self.active[-1]['index'] if self.active else None,
                     status='running', inputs=describe(dict(args=args, kwargs=kwargs)),
                     outputs=None, before=memory, after=None, seconds=None,
                     peak_allocated_bytes=memory['allocated_bytes'],
                     peak_reserved_bytes=memory['reserved_bytes'])
        self.result['entries'].append(entry)
        self.active.append(entry)
        entry['_started'] = time.perf_counter()
        return entry

    def _finish(self, entry, output):
        memory = self._sample()
        entry.update(after=memory, seconds=time.perf_counter()-entry.pop('_started'),
                     outputs=describe(output), status='passed')
        finished = self.active.pop()
        if finished is not entry:
            raise RuntimeError('forward profiling scope mismatch')
        self.result['last_successful_' + entry['kind']] = entry['index']

    def _failure(self, exc):
        # Called at the operator boundary BEFORE Python unwinds model locals.
        # Never synchronize after failure or mask the original exception.
        if self.result['failure'] is not None:
            return
        failure = dict(error_type=type(exc).__name__, error=str(exc),
                       oom=isinstance(exc, torch.OutOfMemoryError),
                       active_entries=[entry['index'] for entry in self.active],
                       traceback=traceback.format_exc())
        try:
            failure['memory'] = self._sample(failed=True)
        except Exception as memory_error:
            failure['memory_error'] = str(memory_error)
        self.result.update(status='failed', failure=failure)
        for entry in self.active:
            entry.update(status='failed', after=failure.get('memory'),
                         seconds=time.perf_counter()-entry.pop('_started'))
        self.save()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        try:
            entry = self._start('operator', str(func), args, kwargs)
            output = func(*args, **kwargs)
            self._finish(entry, output)
            return output
        except Exception as exc:
            self._failure(exc)
            raise

    def __enter__(self):
        try:
            for name, module in self.model.named_modules():
                def before(module, args, kwargs, name=name):
                    self._start('module', name or '<segmentor>', args, kwargs)

                def after(module, args, kwargs, output):
                    self._finish(self.active[-1], output)
                    self.save()

                self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                self.handles.append(module.register_forward_hook(after, with_kwargs=True))
            return super().__enter__()
        except Exception:
            for handle in self.handles:
                handle.remove()
            raise

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self._failure(exc)
            else:
                self.result['status'] = 'passed'
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.active.clear()
            super().__exit__(exc_type, exc, tb)
            self.save()
