"""Read-only MRI whole-volume input, with explicit cohort/label semantics."""
import gzip
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
import torch

from .full_scan import FullScanSample


MODALITIES = ('t1n','t1c','t2w','t2f')  # T1, T1ce, T2, FLAIR, fixed order.
LEGACY_SUFFIXES = dict(t1n='t1',t1c='t1ce',t2w='t2',t2f='flair',seg='seg')


def index_cases(source, *, cohort):
    """Central-directory/file-list metadata only. One cohort per experiment."""
    source = Path(source).expanduser().resolve()
    if cohort not in ('brats2023_gli','brats2024_gli_post','brats2018','brats2019','brats2020','brats2021'):
        raise ValueError('unknown cohort; do not infer label semantics from numeric values')
    if source.is_file() and source.suffix == '.zip':
        with zipfile.ZipFile(source) as archive:
            entries = [i for i in archive.infolist() if i.filename.endswith('.nii.gz')]
            if len({i.filename for i in entries}) != len(entries):
                raise ValueError('duplicate archive member')
            infos = {i.filename: dict(size=i.file_size,crc32=i.CRC) for i in entries}
    elif source.is_dir():
        infos = {p.relative_to(source).as_posix():dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
                 for p in source.rglob('*.nii.gz')}
    else:
        raise ValueError('source must be an existing zip or extracted directory')
    # Known release names can contradict a declaration; never reinterpret RC as ET.
    release_names = [source.name.lower(), *(PurePosixPath(p).parts[0].lower() for p in infos)]
    if any('brats2024' in n or n in ('training_data1_v2','training_data_additional') for n in release_names):
        if cohort != 'brats2024_gli_post':
            raise ValueError('known 2024 postoperative source conflicts with declared cohort')
    if any('brats2023' in n for n in release_names) and cohort != 'brats2023_gli':
        raise ValueError('known 2023 source conflicts with declared cohort')
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
            if parts.name == case+separator+suffix+'.nii.gz':
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
    return dict(cohort=cohort,source=str(source),records=records)


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


class BraTSDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, *, normalization, normalization_epsilon):
        if manifest['cohort'] == 'brats2024_gli_post':
            raise ValueError('2024 post-treatment/RC requires a separately agreed protocol; cannot use three-region preoperative labels')
        if normalization != 'nonzero_zscore' or normalization_epsilon <= 0 or not np.isfinite(normalization_epsilon):
            raise ValueError('explicit per-modality nonzero_zscore with positive epsilon required')
        self.manifest, self.source = manifest, Path(manifest['source'])
        self.epsilon = normalization_epsilon
        self.encoding = '0123' if manifest['cohort'] == 'brats2023_gli' else '0124_preoperative'

    def __len__(self):
        return len(self.manifest['records'])

    def _read(self, record, role):
        import nibabel as nib
        name = record['files'][role]['path']
        if self.source.is_file():
            with zipfile.ZipFile(self.source) as archive:
                return nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(name)))
        path = (self.source/name).resolve()
        if not path.is_relative_to(self.source.resolve()):
            raise ValueError('source path escapes dataset')
        return nib.load(str(path))

    def __getitem__(self, index):
        record = self.manifest['records'][index]
        images = []; reference = None; normalization = []
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
        image = torch.from_numpy(np.stack(images))
        label = torch.from_numpy(label.transpose(2,1,0).copy())
        return FullScanSample(image,label,dict(case_id=record['case_id'],subject_id=record['subject_id'],
            shape_dhw=list(label.shape),affine_xyz=reference[1].tolist(),normalization=normalization,
            modalities=list(MODALITIES),cohort=self.manifest['cohort'],label_encoding=self.encoding))


def validate_partition(manifest, split):
    """Explicit patient-group partition; never split timepoints across roles."""
    from organ_relation.training.state import digest
    if split.get('manifest_hash') != digest(manifest):
        raise ValueError('split manifest hash mismatch')
    records = {r['case_id']:r for r in manifest['records']}
    train, dev = split.get('train',[]), split.get('dev',[])
    if not train or not dev or len(set(train+dev)) != len(train+dev) or set(train+dev) != set(records):
        raise ValueError('train/dev must be nonempty, disjoint, exhaustive case lists')
    if {records[c]['subject_id'] for c in train} & {records[c]['subject_id'] for c in dev}:
        raise ValueError('patient/timepoint leakage across train/dev')
    return ([records[c] for c in train],[records[c] for c in dev])
