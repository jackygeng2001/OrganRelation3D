"""One audited checkpoint/console-cadence migration; not a provenance bypass."""
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .state import digest


SOURCE_COMMIT = 'b4e37bfadf7254fc4869fbb0421ad4706e1f261d'
# Canonical digest of the 50 source hashes in this release's Linux Git blobs.
SOURCE_HASHES = '34e4d8109ff7adc6adec8c6abad6545a891ebbdf392f3bdc151fbe54b0e2acfe'
MIGRATION_TYPE = 'engineering_checkpoint_cadence_only'
CADENCE_FIELDS = ('checkpoint_every_steps', 'console_every_steps')
ENGINEERING_FILES = {
    'scripts/train.py', 'scripts/run_tests.py',
    'src/organ_relation/training/console.py', 'src/organ_relation/training/engine.py',
    'src/organ_relation/training/state.py', 'src/organ_relation/training/extension.py',
    'src/organ_relation/training/resume_migration.py',
}
# Pin the reviewed destination implementations too: a later engine edit is not
# automatically authorized merely because its path is in the engineering set.
TARGET_HASHES = {
    'scripts/train.py': 'a131213a993a4112b2189be2524064976b5f3c8f87d49f397e1f6ee4a9f00db1',
    'scripts/run_tests.py': 'f4a404801ce050d81775e33abe1803ba1f4773a43f10ec8dd30e6d872713f088',
    'src/organ_relation/training/console.py': '10c1d9c8d748979e8d993077dc96c15d617604c3fb0afbb311dd31dba2b71882',
    'src/organ_relation/training/engine.py': '499b35f7483be763d88398d672bdd0f9126df0897a855d4f40516e77ab5a774c',
    'src/organ_relation/training/state.py': '4f1198a98aba391020fdacbffcb58148bf48d925a29e1a4116df7674559a2a61',
    'src/organ_relation/training/extension.py': '5bf355282e958cfa18d60500eb9fe768c758461b10800567e464940eb431fae1',
}


def check_migration_identity(saved, current):
    """Compare ALL identity fields, except exactly the two added cadence keys.

    Environment, unknown fields and scientific provenance metadata stay strict.
    Source hashes may differ only for the reviewed engineering files.
    """
    old, new = copy.deepcopy(saved), copy.deepcopy(current)
    for identity in (old, new):
        if (identity.get('protocol') != 'development_160_40_v2'
                or identity.get('mode') != 'monai_relation_unet'
                or identity.get('relation', {}).get('learnable_relation_scale') is not True
                or identity.get('preprocessing', {}).get('candidate') != 'A'):
            raise ValueError('engineering migration is only for the audited formal A-spacing gated C run')
    previous, execution = old.pop('provenance'), new.pop('provenance')
    for key in CADENCE_FIELDS:
        if key in old['training'] or new['training'].pop(key, None) != 5:
            raise ValueError('migration requires absent old cadence fields and new cadence=5')
    if old != new:
        changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
        raise ValueError('engineering migration scientific identity mismatch: ' + ', '.join(changed))
    if (set(previous) != {'git', 'source_hashes'} or set(execution) != set(previous)
            or previous['git'] != dict(commit=SOURCE_COMMIT, dirty=False)
            or digest(previous['source_hashes']) != SOURCE_HASHES
            or set(execution['git']) != {'commit', 'dirty'}
            or execution['git']['dirty'] is not False
            or not isinstance(execution['git']['commit'], str)
            or len(execution['git']['commit']) != 40
            or execution['git']['commit'] == SOURCE_COMMIT):
        raise ValueError('unsupported or dirty engineering migration provenance')
    before, after = previous['source_hashes'], execution['source_hashes']
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed or not changed <= ENGINEERING_FILES or not before.keys() <= after.keys():
        raise ValueError('engineering migration source mismatch outside audited files')
    if any(after.get(path) != expected for path, expected in TARGET_HASHES.items()):
        raise ValueError('engineering migration destination is not the reviewed implementation')


def check_source_checkpoint(checkpoint, current):
    if checkpoint.get('engineering_resume_migration'):
        raise ValueError('checkpoint already migrated; use ordinary --resume without migration flag')
    if checkpoint.get('origin_identity', checkpoint['identity']) != checkpoint['identity']:
        raise ValueError('migration requires an original, unextended source run')
    if checkpoint.get('horizon', {}).get('extensions'):
        raise ValueError('engineering migration cannot combine with horizon extension')
    check_migration_identity(checkpoint['identity'], current)


def migrate_checkpoint(checkpoint, current, path):
    """Return an in-memory identity upgrade; no training state or file mutation."""
    check_source_checkpoint(checkpoint, current)
    origin = copy.deepcopy(checkpoint['identity'])
    record = dict(
        migration_type=MIGRATION_TYPE, original_checkpoint_commit=origin['provenance']['git']['commit'],
        current_code_commit=current['provenance']['git']['commit'],
        global_step=checkpoint['progress']['global_step'], epoch=checkpoint['progress']['epoch'],
        cursor=checkpoint['progress']['cursor'],
        original_identity_sha256=digest(origin), current_identity_sha256=digest(current),
        parent_checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        utc_time=datetime.now(timezone.utc).isoformat(),
        engineering_changes={k: {'old': None, 'new': current['training'][k]} for k in CADENCE_FIELDS})
    return dict(checkpoint, identity=copy.deepcopy(current), origin_identity=origin,
                engineering_resume_migration=record)


def horizon_checkpoint_view(checkpoint, current, case_count):
    """Preserve immutable run/ledger origin while checking the unchanged horizon.

    This does not waive ordinary resume equality: load_checkpoint checks the
    current identity first. The old identity stays in every stored checkpoint.
    """
    record = checkpoint['engineering_resume_migration']
    origin = checkpoint['origin_identity']
    check_migration_identity(origin, current)
    if (record['migration_type'] != MIGRATION_TYPE
            or record['original_checkpoint_commit'] != SOURCE_COMMIT
            or record['current_code_commit'] != current['provenance']['git']['commit']
            or record['original_identity_sha256'] != digest(origin)
            or record['current_identity_sha256'] != digest(current)
            or record['engineering_changes'] != {k: {'old': None, 'new': 5} for k in CADENCE_FIELDS}
            or not 0 < record['global_step'] <= checkpoint['progress']['global_step']
            or record['epoch'] > checkpoint['progress']['epoch']
            or not 0 <= record['cursor'] < case_count
            or record['global_step'] != record['epoch'] * case_count + record['cursor']
            or checkpoint.get('horizon', {}).get('extensions')):
        raise ValueError('invalid engineering resume migration audit')
    # resolve_horizon needs a same-version origin for its strict horizon check;
    # the actual run origin is never replaced on disk or in the Trainer.
    return dict(checkpoint, origin_identity=current)
