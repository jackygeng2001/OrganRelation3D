"""Frozen development protocol checks and scalar-only epoch summaries."""
import hashlib
import math

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
        result[branch] = dict(loss=sum(v['loss'] for v in values) / len(values))
        if 'hard' in values[0]:
            result[branch]['hard'] = summarize_dice([v['hard'] for v in values])
        if 'soft_per_organ' in values[0]:
            soft = [sum(v['soft_per_organ'][i] for v in values) / len(values) for i in range(15)]
            result[branch].update(soft_mean=sum(soft) / 15, soft_per_organ=soft)
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
    if row['phase'] == 'train' and row.get('relation_scale') and row.get('observation_profile') != 'development_v2':
        return relation_scalars(row['relation_scale'])
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
        if 'hard' in values:
            result[f'Dice/{split}_{title}_Hard'] = values['hard']['mean_case_dice']
        if 'soft_mean' in values:
            result[f'Dice/{split}_{title}_Soft'] = values['soft_mean']
        if split == 'Val':
            for organ in values['hard']['organs']:
                result[f'Dice_Per_Class_Val_{title}/Class_{organ["label"]:02d}'] = organ['mean_dice']
    if row['phase'] == 'train_epoch' and row.get('relation_scale'):
        result.update(relation_scalars(row['relation_scale']))
    return {k: v for k, v in result.items() if v is not None}


def relation_scalars(stats):
    return {'Relation/Gamma': stats['gamma'],
            'Relation/WritebackToFeatureNorm': stats['scaled_writeback_to_feature_norm']}


def validate_early_stopping(options):
    c = options.get('early_stopping')
    if c is None:
        return
    if (set(c) != {'min_epochs', 'patience_epochs', 'min_delta', 'monitor'}
            or options.get('cadence_unit') != 'epoch' or options['validation_role'] != 'internal_dev'
            or options['validation_every'] < 1 or options['max_epochs'] is None
            or c['monitor'] != 'dev_mean_foreground_hard_dice'
            or type(c['min_epochs']) is not int or not 1 <= c['min_epochs'] <= options['max_epochs']
            or type(c['patience_epochs']) is not int or c['patience_epochs'] < 1
            or c['patience_epochs'] % options['validation_every']
            or isinstance(c['min_delta'], bool) or not math.isfinite(c['min_delta']) or c['min_delta'] < 0):
        raise ValueError('invalid early stopping protocol')


def new_early_state():
    return dict(best_metric=None, best_epoch=None, patience_reference_metric=None, no_improvement_count=0,
                last_validation_epoch=0, stopped=False)


def advance_early_stopping(state, score, epoch, options):
    """Track raw best separately from the significant-improvement reference.

    Count validations even before min_epochs, but forbid stopping until then.
    Small records update raw best, but leave the reference unchanged so their
    cumulative improvement can reach min_delta. Equality meets the threshold.
    """
    config = options['early_stopping']
    interval = options['validation_every']
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('early stopping requires finite full-dev Dice')
    if state['stopped'] or epoch != state['last_validation_epoch'] + interval:
        raise ValueError('early stopping validation history is not consecutive')
    best = state['best_metric']
    reference = state['patience_reference_metric']
    significant = reference is None or score >= reference + config['min_delta']
    result = dict(state, last_validation_epoch=epoch,
                  no_improvement_count=0 if significant else state['no_improvement_count'] + 1)
    if significant:
        result['patience_reference_metric'] = score
    if best is None or score > best:
        result.update(best_metric=score, best_epoch=epoch)
    result['stopped'] = (epoch >= config['min_epochs'] and
                        result['no_improvement_count'] * interval >= config['patience_epochs'])
    return result
