"""Exact dense Conv3d via bounded batches of native NCHW Conv2d.

This is a convolution execution backend, not a patch model. Every depth-kernel
plane contributes with the original stride/padding/dilation; all spatial output
positions are assembled before the next network layer sees the result. Only the
original input and weight are saved. No network forward is recomputed backward.
"""
import torch
from torch import nn
from torch.nn import functional as F

DEPTH_CHUNK = 4


def _depth_slice(depth, kernel_index, stride, padding, dilation, start, end):
    # Output j reads input j*stride-padding+kernel_index*dilation.
    # Out-of-image contributions are zero, exactly as zero-padded Conv3d.
    shift = kernel_index * dilation - padding
    lo = max(start, (-shift + stride - 1) // stride)
    hi = min(end, (depth - 1 - shift) // stride + 1)
    return lo, hi, slice(lo * stride + shift, (hi - 1) * stride + shift + 1, stride)


def _planes(x, depth_slice):
    part = x[:, :, depth_slice]
    batch, channels, depth, height, width = part.shape
    # Deliberately NCHW: NHWC can choose the same slow naive MIOpen solvers.
    return part.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width).contiguous()


def _volume(x, batch, depth):
    return x.reshape(batch, depth, x.shape[1], *x.shape[2:]).permute(0, 2, 1, 3, 4)


class _DepthSlicedConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, chunk):
        batch, _, depth, _, _ = x.shape
        size = tuple((n + 2*p - di*(k-1) - 1)//s + 1
                     for n, p, di, k, s in zip(x.shape[2:], padding, dilation, weight.shape[2:], stride))
        if min(size) < 1:
            raise ValueError('convolution kernel exceeds padded input')
        output = torch.empty((batch, weight.shape[0], *size), device=x.device, dtype=x.dtype,
                             memory_format=torch.channels_last_3d).zero_()
        planes = [weight[:, :, k].contiguous() for k in range(weight.shape[2])]
        for start in range(0, size[0], chunk):
            for k, kernel in enumerate(planes):
                lo, hi, source = _depth_slice(depth, k, stride[0], padding[0], dilation[0],
                                             start, min(start + chunk, size[0]))
                if hi <= lo:
                    continue
                part = F.conv2d(_planes(x, source), kernel, stride=stride[1:],
                                padding=padding[1:], dilation=dilation[1:])
                output[:, :, lo:hi].add_(_volume(part, batch, hi-lo))
        if bias is not None:
            output.add_(bias.view(1, -1, 1, 1, 1))
        ctx.save_for_backward(x, weight)
        ctx.options = stride, padding, dilation, chunk
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        stride, padding, dilation, chunk = ctx.options
        batch, _, depth, _, _ = x.shape
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        planes = [weight[:, :, k].contiguous() for k in range(weight.shape[2])]
        if dx is not None or dw is not None:
            for start in range(0, grad_output.shape[2], chunk):
                for k, kernel in enumerate(planes):
                    lo, hi, source = _depth_slice(depth, k, stride[0], padding[0], dilation[0],
                                                 start, min(start + chunk, grad_output.shape[2]))
                    if hi <= lo:
                        continue
                    image = _planes(x, source)
                    gradient = _planes(grad_output, slice(lo, hi))
                    # The same ATen operator used by torch.nn.grad.conv2d_*;
                    # request both derivatives together, without recomputation.
                    gi, gw, _ = torch.ops.aten.convolution_backward.default(
                        gradient, image, kernel, None, list(stride[1:]), list(padding[1:]),
                        list(dilation[1:]), False, [0, 0], 1,
                        (dx is not None, dw is not None, False))
                    if dx is not None:
                        dx[:, :, source].add_(_volume(gi, batch, hi-lo))
                    if dw is not None:
                        dw[:, :, k].add_(gw)
        db = (grad_output.sum((0, 2, 3, 4))
              if ctx.has_bias and ctx.needs_input_grad[2] else None)
        return dx, dw, db, None, None, None, None


class DepthSlicedConv3d(nn.Conv3d):
    """Opt-in FP32/FP64, groups=1, zero-padding Conv3d with unchanged weights."""

    @classmethod
    def from_native(cls, module):
        if (module.groups != 1 or module.padding_mode != 'zeros'
                or isinstance(module.padding, str)):
            raise ValueError('depth_sliced requires groups=1 and numeric zero padding')
        # Do not call Conv3d.__init__: retain Parameter objects, RNG, names and
        # initialization exactly, including an existing optimizer's references.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                     'dilation', 'groups', 'padding_mode', '_reversed_padding_repeated_twice'):
            setattr(result, name, getattr(module, name))
        result.transposed, result.output_padding = False, (0, 0, 0)
        result.register_parameter('weight', module.weight)
        result.register_parameter('bias', module.bias)
        result.train(module.training)
        return result

    def forward(self, x):
        if self.groups != 1 or self.padding_mode != 'zeros' or isinstance(self.padding, str):
            raise ValueError('depth_sliced requires groups=1 and numeric zero padding')
        if (x.ndim != 5 or x.dtype not in (torch.float32, torch.float64)
                or torch.is_autocast_enabled(x.device.type)):
            raise ValueError('depth_sliced requires batched FP32/FP64 input and autocast disabled')
        if self.kernel_size == (1, 1, 1) and self.stride == (1, 1, 1) and self.padding == (0, 0, 0):
            # A pointwise convolution is a channel-wise linear map.
            return F.linear(x.movedim(1, -1), self.weight[:, :, 0, 0, 0], self.bias).movedim(-1, 1)
        return _DepthSlicedConv.apply(x, self.weight, self.bias, self.stride,
                                     self.padding, self.dilation, DEPTH_CHUNK)


def enable_depth_sliced_convolutions(module):
    """Configure a newly built BraTS model, before adding any runtime hooks."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
                raise ValueError('configure convolution execution before registering hooks')
            setattr(module, name, DepthSlicedConv3d.from_native(child))
        else:
            enable_depth_sliced_convolutions(child)
