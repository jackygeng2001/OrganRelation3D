"""Explicit source roles and frozen development artifacts; no implicit test split."""
import json
from pathlib import Path
import random

from .ct_stats import contained_path, select_training
from ..provenance import sha256
from ..training.state import atomic_json, digest

SOURCES = {'official_training', 'official_validation', 'official_test'}
ROLES = {'development_train', 'internal_dev', 'final_train', 'official_validation', 'official_test'}


def training_manifest(root, selection):
    records, _, _ = select_training(Path(root), selection)
    return dict(source_manifest_sha256=sha256(contained_path(Path(root), selection['manifest'])),
                records=[dict(row, source='official_training', modality='CT', verified=True,
                              original_filename=Path(row['image']).name) for row in records])


def validate_artifact(artifact):
    if artifact.get('schema_version') != 1:
        raise ValueError('unsupported split schema')
    records = artifact['manifest']['records']
    ids = [r['case_id'] for r in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('manifest case IDs must be nonempty and unique')
    for row in records:
        if row.get('source') not in SOURCES or row.get('modality') != 'CT' or row.get('verified') is not True:
            raise ValueError('unverified source or non-CT case')
        if row['source'] != 'official_test' and not row.get('label'):
            raise ValueError('supervised source requires labels')
        if row['source'] == 'official_test' and row.get('label') is not None:
            raise ValueError('official test manifest must be image-only')
        for key in ('image', 'label'):
            value = row.get(key)
            if value is not None and (not isinstance(value, str) or Path(value).is_absolute()
                                      or '..' in Path(value).parts or '\\' in value or ':' in value):
                raise ValueError('manifest requires portable relative paths')
        if not row.get('image') or row.get('original_filename') != Path(row['image']).name:
            raise ValueError('original filename must match image path')
    pool = {r['case_id'] for r in records if r['source'] == 'official_training'}
    split = artifact['development']
    train, dev = split['train'], split['internal_dev']
    if (len(train) != len(set(train)) or len(dev) != len(set(dev)) or set(train) & set(dev)
            or set(train) | set(dev) != pool or (pool and (not train or not dev))):
        raise ValueError('development lists must partition official training exactly')
    # If patient grouping is supplied, never silently split the same patient.
    patients = {r['case_id']: r.get('patient_id') for r in records}
    if {patients[c] for c in train if patients[c]} & {patients[c] for c in dev if patients[c]}:
        raise ValueError('patient overlap between development train and internal-dev')
    if artifact.get('manifest_hash') != digest(artifact['manifest']) or artifact.get('split_hash') != digest(split):
        raise ValueError('split/manifest hash mismatch')
    return artifact


def create_development(manifest, seed, train_count):
    ids = sorted(r['case_id'] for r in manifest['records'] if r['source'] == 'official_training')
    if not 0 < train_count < len(ids):
        raise ValueError('both development partitions must be nonempty')
    random.Random(seed).shuffle(ids)
    split = dict(seed=seed, train=ids[:train_count], internal_dev=ids[train_count:])
    artifact = dict(schema_version=1, manifest=manifest, development=split,
                    manifest_hash=digest(manifest), split_hash=digest(split))
    return validate_artifact(artifact)


def write_split(path, artifact):
    validate_artifact(artifact)
    if Path(path).exists():
        raise ValueError('split exists; refusing to regenerate or overwrite it')
    atomic_json(path, artifact)


def load_split(path):
    return validate_artifact(json.loads(Path(path).read_text(encoding='utf-8')))


def select_cases(artifact, role, case_ids=None, limit=None):
    validate_artifact(artifact)
    if role not in ROLES:
        raise ValueError('unknown experiment role')
    records = {r['case_id']: r for r in artifact['manifest']['records']}
    if role in ('development_train', 'internal_dev'):
        allowed = artifact['development']['train' if role == 'development_train' else 'internal_dev']
    else:
        source = 'official_training' if role == 'final_train' else role
        allowed = sorted(c for c, r in records.items() if r['source'] == source)
    chosen = list(case_ids) if case_ids is not None else list(allowed)
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError('subset limit must be positive')
        chosen = chosen[:limit]
    if not chosen or len(chosen) != len(set(chosen)) or not set(chosen) <= set(allowed):
        raise ValueError('case selection violates role or is empty/duplicated')
    if role == 'final_train' and set(chosen) != set(allowed):
        raise ValueError('final training must use the complete official training pool')
    return [records[c] for c in chosen]
