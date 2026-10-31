"""Layout-preserving whole-volume GroupNorm with bounded backward temporaries."""
import torch
from torch import nn

_BACKWARD_CHUNK_ELEMENTS = 1 << 22  # 16 MiB per FP32 elementwise temporary.


class _GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, groups, eps):
        batch, channels = x.shape[:2]
        per_group = channels // groups
        variance, mean = torch.var_mean(x.unflatten(1, (groups, per_group)),
                                       dim=(2,3,4,5), unbiased=False)
        inverse_std = (variance + eps).rsqrt()
        scale = inverse_std.unsqueeze(-1).expand(batch, groups, per_group)
        if weight is not None:
            scale = scale * weight.view(1, groups, per_group)
        shift = -mean.unsqueeze(-1) * scale
        if bias is not None:
            shift = shift + bias.view(1, groups, per_group)
        y = torch.addcmul(shift.reshape(batch, channels, 1, 1, 1), x,
                         scale.reshape(batch, channels, 1, 1, 1))
        ctx.groups, ctx.has_bias = groups, bias is not None
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(x, weight, mean, inverse_std)
        # Exposing statistics as auxiliary outputs preserves their dependence on
        # x for double backward. The module returns only y to its callers.
        return y, mean, inverse_std

    @staticmethod
    def backward(ctx, grad_y, grad_mean, grad_inverse_std):
        x, weight, mean, inverse_std = ctx.saved_tensors
        batch, channels = x.shape[:2]
        groups, per_group = ctx.groups, channels // ctx.groups
        spatial = x.numel() // (batch * channels)
        count = per_group * spatial
        # Both channels-last x and NCDHW/NDHWC upstream gradients can be viewed
        # as [B,spatial,C] without a whole-volume layout conversion.
        flat_x = x.movedim(1,-1).reshape(batch, spatial, channels)
        flat_dy = None if grad_y is None else grad_y.movedim(1,-1).reshape(batch, spatial, channels)
        chunk = max(1, _BACKWARD_CHUNK_ELEMENTS // (batch * channels))
        db = x.new_zeros(batch, channels)
        ds = x.new_zeros(batch, channels)
        if flat_dy is not None:
            db = flat_dy.sum(1)
            for start in range(0, spatial, chunk):
                end = min(spatial, start + chunk)
                ds = ds + (flat_dy[:,start:end] * flat_x[:,start:end]).sum(1)
        gamma = x.new_ones(channels) if weight is None else weight
        sum_dy = (db * gamma).reshape(batch,groups,per_group).sum(2)
        sum_dyx = (ds * gamma).reshape(batch,groups,per_group).sum(2)
        # Exact GN derivative dx = a*dy + b*x + c, where count is the
        # entire group's size. Only these small coefficients are broadcast.
        a = (inverse_std.unsqueeze(-1) * gamma.reshape(1,groups,per_group)).reshape(batch,1,channels)
        b = (mean * sum_dy - sum_dyx) * inverse_std.pow(3) / count
        c = -mean * b - sum_dy * inverse_std / count
        # Auxiliary-output derivatives are zero in ordinary training, but are
        # necessary when autograd differentiates this backward itself.
        if grad_inverse_std is not None:
            extra = -grad_inverse_std * inverse_std.pow(3) / count
            b = b + extra
            c = c - mean * extra
        if grad_mean is not None:
            c = c + grad_mean / count
        dx = None
        if ctx.needs_input_grad[0]:
            b = b.unsqueeze(-1).expand(batch,groups,per_group).reshape(batch,1,channels)
            c = c.unsqueeze(-1).expand(batch,groups,per_group).reshape(batch,1,channels)
            dx = torch.empty_like(x, memory_format=torch.channels_last_3d)
            flat_dx = dx.movedim(1,-1).view(batch,spatial,channels)
            for start in range(0, spatial, chunk):
                end = min(spatial, start + chunk)
                partial = c if flat_dy is None else torch.addcmul(c,flat_dy[:,start:end],a)
                flat_dx[:,start:end] = torch.addcmul(partial,flat_x[:,start:end],b)
        dw = None
        if weight is not None and ctx.needs_input_grad[1]:
            centered = ds.reshape(batch,groups,per_group) - mean.unsqueeze(-1)*db.reshape(batch,groups,per_group)
            dw = (centered * inverse_std.unsqueeze(-1)).sum(0).reshape(channels)
        dbias = db.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, dbias, None, None


class ChannelsLastGroupNorm(nn.GroupNorm):
    """Same groups, population variance, epsilon and affine as nn.GroupNorm.

    Only FP32/FP64 channels-last 5D volumes use the layout-preserving path.
    Backward chunks bound elementwise temporary size, never normalization scope:
    all group statistics and gradient reductions cover the entire volume.
    """

    def forward(self, x):
        if (x.ndim != 5 or not x.is_contiguous(memory_format=torch.channels_last_3d)
                or x.is_contiguous() or x.dtype not in (torch.float32, torch.float64)
                or x.shape[1] != self.num_channels or x.numel() // self.num_groups <= 1
                or torch.is_autocast_enabled(x.device.type)
                or any(p.dtype != x.dtype or p.device != x.device for p in self.parameters())):
            return super().forward(x)
        return _GroupNorm.apply(x,self.weight,self.bias,self.num_groups,self.eps)[0]
