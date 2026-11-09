"""按 epoch/step 的执行节奏与早停状态机（纯函数，不读写文件）。"""
import math


def cadence_due(options, name, step, epoch, epoch_complete):
    interval = options[name]
    return bool(interval and (epoch_complete and epoch % interval == 0
                if options.get('cadence_unit', 'step') == 'epoch' else step % interval == 0))


def validate_early_stopping(options):
    c = options.get('early_stopping')
    if c is None:
        return
    if (set(c) != {'min_epochs', 'patience_epochs', 'min_delta', 'monitor'}
            or options.get('cadence_unit') != 'epoch' or options['validation_role'] != 'internal_dev'
            or options['validation_every'] < 1 or options['max_epochs'] is None
            or c['monitor'] not in ('dev_mean_foreground_hard_dice', 'dev_mean_region_hard_dice')
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
