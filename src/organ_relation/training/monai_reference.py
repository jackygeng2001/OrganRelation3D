"""MONAI loss/monitor adapter and one-backward engineering preflight."""
import json
import time

import torch
from torch import nn

from organ_relation.losses import BranchLoss, FinalLossResult
from organ_relation.models.monai_reference import padding_geometry


class MonaiReferenceLoss(nn.Module):
    """Backpropagate the official DiceCELoss; detached statistics are observers."""
    def __init__(self, constructor):
        super().__init__()
        from monai.losses import DiceCELoss, DiceLoss
        self.constructor = dict(constructor)
        self.loss = DiceCELoss(**constructor)
        dice_args = {k: v for k, v in constructor.items()
                     if k not in ('lambda_dice', 'lambda_ce', 'label_smoothing')}
        dice_args['reduction'] = 'none'
        self.dice_per_class = DiceLoss(**dice_args)

    def forward(self, logits, label):
        if (logits.dtype != torch.float32 or label.dtype != torch.int64
                or tuple(label.shape) != (logits.shape[0], *logits.shape[2:])
                or logits.shape[0] != 1 or logits.shape[1] != 16
                or label.device != logits.device):
            raise ValueError('MONAI reference loss requires matching FP32 logits/int64 labels, batch=1')
        if torch.is_autocast_enabled(logits.device.type) or ((label < 0) | (label > 15)).any():
            raise ValueError('autocast disabled and labels in 0..15 required')
        target = label.unsqueeze(1)  # MONAI expects [B,1,D,H,W]; no padding of GT.
        total = self.loss(logits, target)
        if not torch.isfinite(total):
            raise ValueError('nonfinite MONAI loss')
        with torch.no_grad():
            # Official MONAI implementations, not the project's custom CE/Dice.
            per_class_loss = self.dice_per_class(logits, target).reshape(1, 15)
            per_class = 1 - per_class_loss
            dice = per_class_loss.mean(1)
            ce = self.loss.ce(logits, target).reshape(1)
        return FinalLossResult(total, total.reshape(1), BranchLoss(ce, per_class, dice, total.reshape(1)))


def reference_diagnostics(logits, label, criterion, hard_metrics):
    result = criterion(logits, label)
    probabilities = logits.softmax(1)
    foreground = label > 0
    matched = probabilities.gather(1, label.unsqueeze(1)).squeeze(1)
    present = bool(foreground.any())
    return dict(total_loss=result.total.item(), ce_loss=result.final.ce.item(),
                dice_loss=result.final.dice_loss.item(),
                final_soft_dice=result.final.dice_per_class.mean().item(),
                soft_dice_per_organ=result.final.dice_per_class[0].tolist(),
                predicted_foreground_voxels=sum(o['predicted_voxels'] for o in hard_metrics['organs']),
                foreground_true_positive_voxels=sum(o['true_positive'] for o in hard_metrics['organs']),
                gt_foreground_voxels=int(foreground.sum()),
                gt_foreground_true_class_mean_probability=matched[foreground].mean().item() if present else None,
                gt_foreground_background_mean_probability=probabilities[:, 0][foreground].mean().item() if present else None)


def preflight_backward(model, criterion, dataset, identity):
    """No Trainer/run/optimizer writes or optimizer.step; fail once, never retry."""
    device = next(model.parameters()).device
    stages = {}
    active = 'preprocess_transfer'
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def measure(name, operation):
        nonlocal active
        active = name
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        value = operation()
        sync()
        stages[name] = dict(seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None)
        print(name, json.dumps(stages[name]), flush=True)
        return value
    try:
        sample = dataset[0]
        image, label = sample.image.unsqueeze(0).to(device), sample.label.unsqueeze(0).to(device)
        if identity['training']['memory_format'] == 'channels_last_3d':
            image = image.contiguous(memory_format=torch.channels_last_3d)
        print('reference_preflight', json.dumps(dict(mode=identity['mode'],
            geometry=padding_geometry(image.shape[2:]), parameter_count=identity['parameter_count'],
            model=identity['model'], loss=identity['loss'], preprocessing=identity['preprocessing'],
            environment=identity['environment'], provenance=identity['provenance']['git'])), flush=True)
        output = measure('forward', lambda: model(image))
        loss = measure('loss', lambda: criterion(output.final_logits, label))
        measure('backward', loss.total.backward)
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        nonfinite = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        zeros = [n for n, p in model.named_parameters() if p.grad is not None and not p.grad.any()]
        print('preflight_result', json.dumps(dict(status='passed' if not missing and not nonfinite else 'failed',
            total_loss=loss.total.item(), ce_loss=loss.final.ce.item(), dice_loss=loss.final.dice_loss.item(),
            missing_gradients=missing, nonfinite_gradients=nonfinite, zero_gradient_tensors=zeros,
            forward_loss_backward_seconds=sum(s['seconds'] for s in stages.values()), optimizer_steps=0)), flush=True)
        if missing or nonfinite:
            raise ValueError('invalid gradients in MONAI reference preflight')
    except (RuntimeError, ValueError) as exc:
        print('preflight_failed', json.dumps(dict(stage=active, error=str(exc), stages=stages)), flush=True)
        raise
