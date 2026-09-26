"""Internal model-grid hard Dice, not AMOS official metrics or JointLoss soft Dice."""
import torch

METRIC_PROTOCOL = 'model_grid_hard_dice_v1_both_empty_null'


def hard_dice(prediction, label):
    if label is None:
        raise ValueError('supervised Dice requires a label; test inference must not call it')
    if prediction.shape != label.shape or prediction.ndim != 3:
        raise ValueError('hard Dice expects one matching [D,H,W] case')
    if prediction.dtype != torch.int64 or label.dtype != torch.int64 or prediction.device != label.device:
        raise ValueError('hard Dice requires int64 indices on the same device')
    if ((prediction < 0) | (prediction > 15) | (label < 0) | (label > 15)).any():
        raise ValueError('labels must be in 0..15')
    confusion = torch.bincount((label.flatten()*16 + prediction.flatten()), minlength=256).reshape(16, 16)
    counts = confusion.cpu().tolist()
    organs = []
    for c in range(1, 16):
        gt = sum(counts[c])
        pred = sum(row[c] for row in counts)
        tp = counts[c][c]
        organs.append(dict(label=c, gt_voxels=gt, predicted_voxels=pred, true_positive=tp,
                           false_positive_voxels=pred-tp,
                           dice=2*tp/(gt+pred) if gt+pred else None))
    values = [o['dice'] for o in organs if o['dice'] is not None]
    return dict(protocol=METRIC_PROTOCOL, organs=organs,
                mean_dice=sum(values)/len(values) if values else None)


def summarize_dice(cases):
    means = [c['mean_dice'] for c in cases if c['mean_dice'] is not None]
    organs = []
    for index in range(15):
        values = [c['organs'][index]['dice'] for c in cases if c['organs'][index]['dice'] is not None]
        organs.append(dict(label=index+1, valid_cases=len(values),
                           mean_dice=sum(values)/len(values) if values else None))
    return dict(protocol=METRIC_PROTOCOL, cases=len(cases), valid_cases=len(means),
                mean_case_dice=sum(means)/len(means) if means else None, organs=organs,
                nsd=None, nsd_status='not_implemented_pending_official_protocol')
