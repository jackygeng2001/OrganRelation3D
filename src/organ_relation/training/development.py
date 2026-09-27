"""Frozen development protocol checks and scalar-only epoch summaries."""
import hashlib

from ..metrics import summarize_dice


def verify_frozen_split(path, artifact, expected_hash):
    """Check the JSON's canonical partition hash, separately record file checksum.

    load_split has already verified internal manifest and partition hashes.
    """
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact['split_hash'] != expected_hash:
        raise ValueError('frozen development SHA256 mismatch')
    split = artifact['development']
    if (len(split['train']), len(split['internal_dev'])) != (160, 40):
        raise ValueError('frozen development requires exactly 160/40 cases')
    return dict(expected_split_hash=expected_hash,
                file_sha256=file_hash, split_hash=artifact['split_hash'])


def summarize_branches(cases):
    """Case means, never voxel pooling; hard Dice keeps existing empty semantics."""
    result = {}
    for branch in ('final', 'coarse'):
        if branch not in cases[0]:
            continue
        values = [case[branch] for case in cases]
        soft = [sum(v['soft_per_organ'][i] for v in values) / len(values) for i in range(15)]
        result[branch] = dict(loss=sum(v['loss'] for v in values) / len(values),
                              hard=summarize_dice([v['hard'] for v in values]),
                              soft_mean=sum(soft) / 15, soft_per_organ=soft)
    result['total_loss'] = sum(c['total_loss'] for c in cases) / len(cases)
    return result


def validation_branches(record):
    diagnostic = record['diagnostic']
    result = dict(final=dict(loss=diagnostic['total_loss'], hard=record,
                             soft_per_organ=diagnostic['soft_dice_per_organ']),
                  total_loss=diagnostic.get('joint_total_loss', diagnostic['total_loss']))
    if 'coarse' in diagnostic:
        c = diagnostic['coarse']
        result['coarse'] = dict(loss=c['total_loss'], hard=c['hard_metrics'],
                                soft_per_organ=c['soft_dice_per_organ'])
    return result


def cadence_due(options, name, step, epoch, epoch_complete):
    interval = options[name]
    return bool(interval and (epoch_complete and epoch % interval == 0
                if options.get('cadence_unit', 'step') == 'epoch' else step % interval == 0))


def development_scalars(row):
    """Strict formal TensorBoard whitelist; diagnostics remain in JSONL only."""
    if row['phase'] not in ('train_epoch', 'internal_dev'):
        return {}
    split = 'Train' if row['phase'] == 'train_epoch' else 'Val'
    summary = row['epoch_metrics']
    result = {}
    if split == 'Train':
        result['Loss/Train_Total'] = summary['total_loss']
    for branch in ('final', 'coarse'):
        if branch not in summary:
            continue
        values = summary[branch]
        title = branch.title()
        result[f'Loss/{split}_{title}'] = values['loss']
        result[f'Dice/{split}_{title}_Hard'] = values['hard']['mean_case_dice']
        result[f'Dice/{split}_{title}_Soft'] = values['soft_mean']
        if split == 'Val':
            for organ in values['hard']['organs']:
                result[f'Dice_Per_Class_Val_{title}/Class_{organ["label"]:02d}'] = organ['mean_dice']
    return {k: v for k, v in result.items() if v is not None}
