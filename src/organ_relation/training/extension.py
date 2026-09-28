"""Explicit step-horizon extension, with immutable scientific/run identity.

Only the audited pre-extension release can cross a source-version boundary.
This is not a general ignore-provenance or config-migration facility.
"""
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .state import digest, identity_with_ce_weights


LEGACY_COMMIT = '88ea4fc9fd8bc2aa3ede2c3fe87b343c386e6941'
LEGACY_SOURCES = 'dca57430d67e2636fad9a20fe9ca3cb0b8ee3573e744665d9cf8a23c332ce19d'
EXTENSION_FILES = {'scripts/train.py', 'src/organ_relation/training/engine.py',
                   'src/organ_relation/training/state.py', 'src/organ_relation/training/extension.py'}


def check_extension_identity(saved, current):
    # Even max_steps in the input config must match. The override lives only
    # in checkpoint.horizon; users cannot smuggle config changes through it.
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    previous, execution = old.pop('provenance'), new.pop('provenance')
    if identity_with_ce_weights(old) != identity_with_ce_weights(new):
        raise ValueError('extension identity mismatch: only explicit horizon increase is allowed')
    if previous == execution:
        return
    if (not isinstance(previous, dict) or not isinstance(execution, dict)
            or previous.get('git') != dict(commit=LEGACY_COMMIT, dirty=False)
            or digest(previous.get('source_hashes')) != LEGACY_SOURCES
            or execution.get('git', {}).get('dirty') is not False
            or not execution.get('git', {}).get('commit')
            or set(previous) != set(execution)):
        raise ValueError('extension provenance mismatch: unsupported source upgrade or dirty checkout')
    before, after = previous['source_hashes'], execution.get('source_hashes', {})
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed or not changed <= EXTENSION_FILES or not before.keys() <= after.keys():
        raise ValueError('extension source mismatch: model/data/loss and other sources must remain identical')


def resolve_horizon(options, case_count, checkpoint, extend_to, identity, resume):
    """Pure validation/planning. No RNG use, checkpoint writes or config edits."""
    if checkpoint and checkpoint.get('engineering_resume_migration'):
        if extend_to is not None:
            raise ValueError('engineering migration cannot combine with extension')
        from .resume_migration import horizon_checkpoint_view
        checkpoint = horizon_checkpoint_view(checkpoint, identity, case_count)
    if extend_to is not None and options.get('early_stopping'):
        raise ValueError('early-stopped/max-epoch protocol cannot be bypassed by horizon extension')
    original = min(v for v in (options['max_steps'],
                   options['max_epochs'] * case_count if options['max_epochs'] else None) if v is not None)
    horizon = copy.deepcopy(checkpoint.get('horizon')) if checkpoint else None
    if horizon is None:
        horizon = dict(original_total_steps=original, total_steps=original, extensions=[])
    if (set(horizon) != {'original_total_steps', 'total_steps', 'extensions'}
            or type(horizon['original_total_steps']) is not int or horizon['original_total_steps'] != original
            or type(horizon['total_steps']) is not int or not isinstance(horizon['extensions'], list)):
        raise ValueError('invalid checkpoint horizon')
    total, previous_step = original, 0
    epoch_mode = options.get('cadence_unit') == 'epoch'
    epoch_limit = options['max_epochs'] * case_count if options['max_epochs'] and (not epoch_mode or options.get('early_stopping')) else None
    origin = checkpoint.get('origin_identity', checkpoint['identity']) if checkpoint else identity
    origin_science, execution_science = copy.deepcopy(origin), copy.deepcopy(identity)
    previous_provenance = origin_science.pop('provenance')
    execution_science.pop('provenance')
    if identity_with_ce_weights(origin_science) != identity_with_ce_weights(execution_science):
        raise ValueError('original run identity mismatch outside provenance')
    for event in horizon['extensions']:
        at, target = event['at_global_step'], event['total_steps']
        if (type(at) is not int or not previous_step <= at <= total or at < 1
                or type(target) is not int or target <= total
                or (not epoch_mode and (options['max_steps'] is None or target <= options['max_steps']))
                or (epoch_mode and target % case_count != 0)
                or (epoch_limit is not None and target > epoch_limit)
                or event['previous_total_steps'] != total
                or event['max_epochs'] != options['max_epochs']
                or event['previous_provenance'] != previous_provenance
                or at > checkpoint['progress']['global_step']):
            raise ValueError('invalid checkpoint extension history')
        total, previous_step = target, at
        previous_provenance = event['execution_provenance']
    if total != horizon['total_steps']:
        raise ValueError('checkpoint horizon does not match extension history')
    if checkpoint and previous_provenance != checkpoint['identity']['provenance']:
        raise ValueError('checkpoint provenance does not match extension history')
    if checkpoint and (type(checkpoint['progress']['global_step']) is not int
                       or not 1 <= checkpoint['progress']['global_step'] <= total):
        raise ValueError('checkpoint progress exceeds its committed horizon')
    if extend_to is not None:
        if checkpoint is None or resume is None:
            raise ValueError('--extend-to requires --resume')
        step = checkpoint['progress']['global_step']
        if (type(extend_to) is not int or (not epoch_mode and options['max_steps'] is None)
                or extend_to <= max(total, options['max_steps'] or original, step)
                or (epoch_mode and extend_to % case_count != 0)):
            raise ValueError('--extend-to must strictly increase planned max_steps and exceed completed global_step')
        if epoch_limit is not None and extend_to > epoch_limit:
            raise ValueError('--extend-to exceeds unchanged max_epochs limit')
        horizon['extensions'].append(dict(
            at_global_step=step, previous_total_steps=total, total_steps=extend_to,
            max_epochs=options['max_epochs'], utc_time=datetime.now(timezone.utc).isoformat(),
            parent_checkpoint_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
            previous_provenance=copy.deepcopy(checkpoint['identity']['provenance']),
            execution_provenance=copy.deepcopy(identity['provenance'])))
        horizon['total_steps'] = extend_to
    return horizon
