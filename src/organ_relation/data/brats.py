"""Read-only MRI whole-volume input, with explicit cohort/label semantics."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import torch

from .full_scan import FullScanSample


MODALITIES = ('t1n','t1c','t2w','t2f')  # T1, T1ce, T2, FLAIR, fixed order.
LEGACY_SUFFIXES = dict(t1n='t1',t1c='t1ce',t2w='t2',t2f='flair',seg='seg')
CACHE_FORMAT = 'brats_npy_v1'  # Bitwise copy of the loader output; never a different preprocessing.


def index_cases(source, *, cohort, partition="all"):
    """Central-directory/file-list metadata only. One cohort per experiment."""
    if partition not in ("all", "training"):
        raise ValueError("source partition must be all or training")
    source = Path(source).expanduser().resolve()
    if cohort not in ('brats2023_gli','brats2024_gli_post','brats2018','brats2019','brats2020','brats2021'):
        raise ValueError('unknown cohort; do not infer label semantics from numeric values')
    if source.is_file() and source.suffix == '.zip':
        with zipfile.ZipFile(source) as archive:
            entries = [i for i in archive.infolist() if i.filename.endswith(('.nii', '.nii.gz'))]
            if len({i.filename for i in entries}) != len(entries):
                raise ValueError('duplicate archive member')
            infos = {i.filename: dict(size=i.file_size,crc32=i.CRC) for i in entries}
    elif source.is_dir():
        infos = {p.relative_to(source).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                 for p in source.rglob('*') if p.is_file() and p.name.endswith(('.nii', '.nii.gz'))}
    else:
        raise ValueError('source must be an existing zip or extracted directory')
    # Known release names can contradict a declaration; never reinterpret RC as ET.
    release_names = [source.name.lower(), *(PurePosixPath(p).parts[0].lower() for p in infos)]
    if any('brats2024' in n or n in ('training_data1_v2','training_data_additional') for n in release_names):
        if cohort != 'brats2024_gli_post':
            raise ValueError('known 2024 postoperative source conflicts with declared cohort')
    if any('brats2023' in n for n in release_names) and cohort != 'brats2023_gli':
        raise ValueError('known 2023 source conflicts with declared cohort')
    for year in (2018, 2019, 2020, 2021):
        if any(re.search(rf'brats[_-]?{year}', n) for n in release_names) and cohort != f'brats{year}':
            raise ValueError('known legacy year conflicts with declared cohort')
    if partition == 'training':
        if cohort not in ('brats2018','brats2019','brats2020'):
            raise ValueError('training subtree selection is only defined for legacy 2018-2020 releases')
        infos = {name: info for name, info in infos.items() if any(
            'brats' in part.lower() and ('data_training' in part.lower() or 'trainingdata' in part.lower())
            for part in PurePosixPath(name).parts[:-1])}
        if not infos:
            raise ValueError('no recognized legacy training subtree; do not infer labels from missing files')
    groups = {}
    for name, fingerprint in sorted(infos.items()):
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts:
            raise ValueError('unsafe source member path')
        if len(parts.parts) < 2:
            continue
        case = parts.parent.name
        suffixes = {k:k for k in (*MODALITIES,'seg')} if cohort.startswith(('brats2023','brats2024')) else LEGACY_SUFFIXES
        for role, suffix in suffixes.items():
            separator = '-' if cohort.startswith(('brats2023','brats2024')) else '_'
            regular = parts.name in (case+separator+suffix+'.nii.gz', case+separator+suffix+'.nii')
            # Audited Kaggle 2020 mirror: exact alias, never a generic '*Segm' guess.
            alias = (cohort == 'brats2020' and case == 'BraTS20_Training_355'
                     and role == 'seg' and parts.name == 'W39_1998.09.19_Segm.nii')
            if regular or alias:
                group = groups.setdefault(case,{})
                if role in group:
                    raise ValueError('duplicate case/modality: '+case+'/'+role)
                group[role] = dict(path=name,**fingerprint)
    records = []
    for case, files in sorted(groups.items()):
        if set(files) != {*MODALITIES,'seg'}:
            raise ValueError('incomplete multimodal labeled case: '+case)
        subject = case.rsplit('-',1)[0] if re.fullmatch(r'BraTS-GLI-\d+-\d+',case) else case
        records.append(dict(case_id=case,subject_id=subject,files=files))
    if not records:
        raise ValueError('no complete labeled BraTS cases')
    result = dict(cohort=cohort,source=str(source),records=records)
    if partition != 'all':
        result['source_partition'] = partition
    return result


def canonical_labels(array, *, encoding):
    if not np.isfinite(array).all() or not np.equal(array,np.round(array)).all():
        raise ValueError('segmentation must contain finite integer label values')
    allowed = (0,1,2,3) if encoding == '0123' else (0,1,2,4) if encoding == '0124_preoperative' else ()
    if not allowed or not np.isin(array,allowed).all():
        raise ValueError('unexpected label encoding; postoperative resection cavity is not legacy ET')
    label = array.astype(np.int64)
    if encoding == '0124_preoperative':
        label[label == 4] = 3
    return label


def validate_spatial_crop(crop):
    """None keeps the full registered grid; nonzero_bbox is the only crop rule."""
    if crop is None:
        return None
    if (not isinstance(crop, dict) or set(crop) != {'kind', 'margin', 'multiple_of'}
            or crop['kind'] != 'nonzero_bbox'
            or type(crop['margin']) is not int or crop['margin'] < 0
            or type(crop['multiple_of']) is not int or crop['multiple_of'] < 1):
        raise ValueError('spatial_crop must be null or {kind: nonzero_bbox, margin>=0, multiple_of>=1}')
    return dict(crop)


def nonzero_bbox(support, *, margin, multiple_of):
    """DHW [start, stop) box of image-only support; never smaller than the support.

    The tight box of all nonzero voxels is widened by `margin`, then grown
    (centred) to a multiple of `multiple_of` where the volume allows, and
    clamped/shifted inside the volume. Every supported voxel stays inside.
    """
    if support.ndim != 3 or support.dtype != np.bool_ or not support.any():
        raise ValueError('nonempty boolean DHW support mask required')
    tight = []
    for axis in range(3):
        index = np.flatnonzero(support.any(axis=tuple(a for a in range(3) if a != axis)))
        tight.append((int(index[0]), int(index[-1]) + 1))
    return expand_box(tight, support.shape, margin=margin, multiple_of=multiple_of)


def expand_box(tight, shape, *, margin, multiple_of):
    """Grow per-axis [low, high) boxes; result always contains the input box."""
    start, stop = [], []
    for (low, high), size in zip(tight, shape):
        low, high = max(low - margin, 0), min(high + margin, size)
        target = min(-(-(high - low) // multiple_of) * multiple_of, size)
        low = max(low - (target - (high - low)) // 2, 0)
        high = low + target
        if high > size:
            low, high = size - target, size
        start.append(low)
        stop.append(high)
    return start, stop


class BraTSDataset(torch.utils.data.Dataset):
    """Loader output is a pure function of the source files and preprocessing.

    `cache_dir` is an execution option only: each case's exact output arrays
    and metadata are stored once as .npy/.json and later read back bitwise.
    The entry key covers cohort, label encoding, every source file fingerprint
    (ZIP size/CRC32 or file size/mtime), the preprocessing settings and this
    loader file's SHA256, so any change creates a new entry instead of reuse.
    """
    def __init__(self, manifest, *, normalization, normalization_epsilon, spatial_crop=None, cache_dir=None):
        if manifest['cohort'] == 'brats2024_gli_post':
            raise ValueError('2024 post-treatment/RC requires a separately agreed protocol; cannot use three-region preoperative labels')
        if normalization != 'nonzero_zscore' or normalization_epsilon <= 0 or not np.isfinite(normalization_epsilon):
            raise ValueError('explicit per-modality nonzero_zscore with positive epsilon required')
        self.manifest, self.source = manifest, Path(manifest['source'])
        self.epsilon = normalization_epsilon
        self.spatial_crop = validate_spatial_crop(spatial_crop)
        self.encoding = '0123' if manifest['cohort'] == 'brats2023_gli' else '0124_preoperative'
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
            source = self.source.expanduser().resolve()
            if self.cache_dir.is_relative_to(source if source.is_dir() else source.parent):
                raise ValueError('data cache must be outside the read-only source dataset')
            self.loader_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def __len__(self):
        return len(self.manifest['records'])

    def _read(self, record, role):
        import nibabel as nib
        name = record['files'][role]['path']
        if self.source.is_file():
            with zipfile.ZipFile(self.source) as archive:
                raw = archive.read(name)
                return nib.Nifti1Image.from_bytes(gzip.decompress(raw) if name.endswith('.gz') else raw)
        path = (self.source/name).resolve()
        if not path.is_relative_to(self.source.resolve()):
            raise ValueError('source path escapes dataset')
        return nib.load(str(path))

    def __getitem__(self, index):
        record = self.manifest['records'][index]
        if self.cache_dir is None:
            image, label, metadata = self._build(record)
        else:
            image, label, metadata = self._cached(record)
        return FullScanSample(torch.from_numpy(np.ascontiguousarray(image)),
                              torch.from_numpy(np.ascontiguousarray(label)), metadata)

    def cache_key(self, record):
        return dict(format=CACHE_FORMAT, cohort=self.manifest['cohort'], label_encoding=self.encoding,
                    case_id=record['case_id'], subject_id=record['subject_id'], files=record['files'],
                    preprocessing=dict(normalization='nonzero_zscore', normalization_epsilon=self.epsilon,
                                       spatial_crop=self.spatial_crop),
                    loader_sha256=self.loader_sha256)

    def _cached(self, record):
        key = self.cache_key(record)
        stem = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        folder = self.cache_dir / self.manifest['cohort'] / record['case_id']
        paths = {name: folder / f'{stem}.{name}' for name in ('image.npy', 'label.npy', 'json')}
        if paths['json'].exists():  # Written last: its presence marks a complete entry.
            entry = json.loads(paths['json'].read_text())
            try:
                image = np.load(paths['image.npy'], allow_pickle=False)
                label = np.load(paths['label.npy'], allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f'unreadable data cache entry; delete {folder}: {exc}') from exc
            if (entry.get('key') != key or image.dtype != np.float32 or label.dtype != np.uint8
                    or list(image.shape) != entry['image_shape'] or list(label.shape) != entry['label_shape']):
                raise ValueError(f'data cache entry does not match its key/shape; delete {folder}')
            return image, label.astype(np.int64), entry['metadata']
        image, label, metadata = self._build(record)
        if label.min() < 0 or label.max() > 255:
            raise ValueError('canonical labels must fit uint8 for caching')
        folder.mkdir(parents=True, exist_ok=True)
        entry = dict(key=key, image_shape=list(image.shape), label_shape=list(label.shape), metadata=metadata)
        for name, array in (('image.npy', np.ascontiguousarray(image)), ('label.npy', label.astype(np.uint8))):
            temporary = folder / f'.{stem}.{name}.{os.getpid()}.tmp'
            with temporary.open('wb') as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary, paths[name])
        temporary = folder / f'.{stem}.json.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(entry))
        os.replace(temporary, paths['json'])
        return image, label, json.loads(json.dumps(metadata))

    def _build(self, record):
        images = []; reference = None; normalization = []; support = None
        for role in (*MODALITIES,'seg'):
            nii = self._read(record,role)
            if len(nii.shape) != 3 or not np.isfinite(nii.affine).all():
                raise ValueError('finite aligned 3D NIfTI volumes required')
            if reference is None:
                reference = (nii.shape,nii.affine.copy())
            elif nii.shape != reference[0] or not np.allclose(nii.affine,reference[1],rtol=0,atol=1e-5):
                raise ValueError('modalities/label grids differ; no silent registration or resampling')
            array = nii.get_fdata(dtype=np.float32)  # Honor per-file slope/intercept; never CT HU transforms.
            if not np.isfinite(array).all():
                raise ValueError('nonfinite MRI volume')
            if role == 'seg':
                label = canonical_labels(array,encoding=self.encoding)
                continue
            mask = array != 0  # Image-only intensity support; spatial coverage unchanged.
            if not mask.any():
                raise ValueError('empty MRI modality: '+role)
            mean, std = float(array[mask].mean(dtype=np.float64)), float(array[mask].std(dtype=np.float64))
            normalized = np.zeros_like(array)
            normalized[mask] = (array[mask]-mean)/max(std,self.epsilon)
            images.append(normalized.transpose(2,1,0).copy())
            normalization.append(dict(modality=role,mean=mean,std=std))
            support = mask.transpose(2,1,0) if support is None else support | mask.transpose(2,1,0)
        image = np.stack(images)
        label = label.transpose(2,1,0)
        metadata = dict(case_id=record['case_id'],subject_id=record['subject_id'],
            affine_xyz=reference[1].tolist(),normalization=normalization,
            modalities=list(MODALITIES),cohort=self.manifest['cohort'],label_encoding=self.encoding)
        if self.spatial_crop is not None:
            # Box from the union of nonzero voxels of the four images only;
            # the label is used solely to refuse a crop that would drop GT.
            start, stop = nonzero_bbox(support, margin=self.spatial_crop['margin'],
                                       multiple_of=self.spatial_crop['multiple_of'])
            window = tuple(slice(a, b) for a, b in zip(start, stop))
            cropped = label[window]
            if np.count_nonzero(cropped) != np.count_nonzero(label):
                raise ValueError('spatial crop would discard labeled tumor voxels: '+record['case_id'])
            metadata['spatial_crop'] = dict(self.spatial_crop, original_shape_dhw=list(label.shape),
                                            start_dhw=start, stop_dhw=stop)
            image, label = image[(slice(None), *window)], cropped
        metadata['shape_dhw'] = list(label.shape)
        return image, label, metadata


def make_pilot_partition(manifest, *, train_subjects, dev_subjects, seed, excluded_subjects=()):
    """Metadata-only, reproducible pilot: one lexicographically first scan per patient.

    Unselected scans remain explicitly unused; this is not a formal cohort split.
    Hash ranking avoids changing any training/global RNG state.
    """
    from organ_relation.training.state import digest
    if manifest.get('cohort') != 'brats2023_gli':
        raise ValueError('patient pilot currently supports brats2023_gli only')
    if any(type(n) is not int or n < 1 for n in (train_subjects, dev_subjects)) or type(seed) is not int:
        raise ValueError('positive integer patient counts and integer seed required')
    groups = {}
    case_ids = []
    for record in manifest['records']:
        case_ids.append(record['case_id'])
        groups.setdefault(record['subject_id'], []).append(record['case_id'])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('duplicate manifest case')
    excluded = sorted(set(excluded_subjects))
    if not set(excluded) <= set(groups):
        raise ValueError('excluded patient absent from manifest')
    ranked = sorted(set(groups)-set(excluded), key=lambda subject: (digest([seed, subject]), subject))
    if train_subjects + dev_subjects > len(ranked):
        raise ValueError('not enough eligible patients for pilot')
    selected = [min(groups[subject]) for subject in ranked[:train_subjects+dev_subjects]]
    return dict(kind='patient_pilot_v1', manifest_hash=digest(manifest), seed=seed,
                train_subjects=train_subjects, dev_subjects=dev_subjects,
                excluded_subjects=excluded, scan_policy='lexicographically_first_per_subject',
                train=selected[:train_subjects], dev=selected[train_subjects:],
                unused=sorted(set(case_ids)-set(selected)))


def make_folds(manifest, *, seed, folds=5):
    """Label-independent patient folds, stable under record ordering; no global RNG."""
    import random
    from organ_relation.training.state import digest
    if manifest['cohort'] not in ('brats2018', 'brats2019', 'brats2020'):
        raise ValueError('legacy year-specific folds only')
    subjects = sorted({r['subject_id'] for r in manifest['records']})
    if type(seed) is not int or type(folds) is not int or not 2 <= folds <= len(subjects):
        raise ValueError('integer seed and valid fold count required')
    random.Random(seed).shuffle(subjects)
    records = sorted(manifest['records'], key=lambda r: r['case_id'])
    result = []
    for fold in range(folds):
        dev = set(subjects[fold::folds])
        result.append(dict(manifest_hash=digest(manifest),
            train=[r['case_id'] for r in records if r['subject_id'] not in dev],
            dev=[r['case_id'] for r in records if r['subject_id'] in dev],
            fold=fold, folds=folds, seed=seed, cohort=manifest['cohort'],
            protocol='patient_kfold_v1', stratification='none'))
    return result


def validate_partition(manifest, split):
    """Explicit patient-group partition; never split timepoints across roles."""
    from organ_relation.training.state import digest
    if split.get('manifest_hash') != digest(manifest):
        raise ValueError('split manifest hash mismatch')
    if split.get('protocol') == 'patient_kfold_v1':
        candidates = make_folds(manifest, seed=split.get('seed'), folds=split.get('folds'))
        fold = split.get('fold')
        if type(fold) is not int or not 0 <= fold < len(candidates) or split != candidates[fold]:
            raise ValueError('fold differs from deterministic patient partition')
    elif 'protocol' in split:
        raise ValueError('unknown partition protocol')
    if split.get('kind') == 'patient_pilot_v1':
        expected = make_pilot_partition(manifest, train_subjects=split.get('train_subjects'),
            dev_subjects=split.get('dev_subjects'), seed=split.get('seed'),
            excluded_subjects=split.get('excluded_subjects', []))
        if split != expected:
            raise ValueError('pilot split differs from recorded deterministic patient selection')
        records = {r['case_id']:r for r in manifest['records']}
        return ([records[c] for c in split['train']], [records[c] for c in split['dev']])
    if split.get('kind') is not None or 'unused' in split:
        raise ValueError('unused cases require explicit patient_pilot_v1 split')
    records = {r['case_id']:r for r in manifest['records']}
    train, dev = split.get('train',[]), split.get('dev',[])
    if not train or not dev or len(set(train+dev)) != len(train+dev) or set(train+dev) != set(records):
        raise ValueError('train/dev must be nonempty, disjoint, exhaustive case lists')
    if {records[c]['subject_id'] for c in train} & {records[c]['subject_id'] for c in dev}:
        raise ValueError('patient/timepoint leakage across train/dev')
    return ([records[c] for c in train],[records[c] for c in dev])
