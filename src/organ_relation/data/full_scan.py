"""Full-scan, batch-one NIfTI-to-Tensor contract, independent of pilot orchestration.

Physical grids use RAS millimetres; tensor spatial indices are (S, A, R).
Only audited cardinal, millimetre NIfTI-1 scans are currently supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from .ct_stats import contained_path, select_training
from .fidelity import affine4, antialias_sigmas, build_target_grid, load_pair, resample
from .nifti_header import compare_geometry, read_header


@dataclass(frozen=True)
class ScanPair:
    case_id: str
    image_path: Path
    label_path: Path


class FullScanSample(NamedTuple):
    image: torch.Tensor  # [1,1,D,H,W], float32 CPU; no label-derived processing.
    label: torch.Tensor  # [1,D,H,W], int64 CPU; supervision only.
    metadata: dict  # Original/target/tensor affines and both index mappings.


class FullScanPreprocessor:
    """Explicit feasibility preprocessing; no crop, pad, ROI or batch collation.

scaled_hu means file-specific scaling ONLY, without intensity normalization.
The geometry/antialias choices are recorded validation settings, not a claim
that formal training preprocessing or the final spacing has been selected.
"""

    def __init__(self, config: dict, candidate: str, *, max_voxels: int | None = None):
        if config.get('grid') != 'ceil_cell_coverage_centered_extension':
            raise ValueError('unsupported grid convention')
        if config.get('boundary') != 'nearest_edge_replication':
            raise ValueError('unsupported boundary convention')
        if config.get('intensity') != 'scaled_hu':
            raise ValueError('only explicit scaled_hu is implemented; normalization needs a separate protocol')
        if type(config.get('antialias')) is not bool:
            raise ValueError('antialias must be explicit boolean')
        truncate = config.get('gaussian_truncate')
        if isinstance(truncate, bool) or not isinstance(truncate, (float, int)) or not math.isfinite(truncate) or truncate <= 0:
            raise ValueError('gaussian_truncate must be finite and positive')
        candidates = config.get('spacing_candidates', {})
        if set(candidates) != {'A', 'B', 'C'} or candidate not in candidates:
            raise ValueError('A/B/C candidates must all be retained; select one of A/B/C')
        for spacing in candidates.values():
            if len(spacing) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in spacing):
                raise ValueError('spacing must be three finite positive values in R,A,S order')
        if max_voxels is not None and (type(max_voxels) is not int or max_voxels < 1):
            raise ValueError('max_voxels must be a positive integer or None')
        self.candidate = candidate
        self.spacing = tuple(candidates[candidate])
        self.antialias = config['antialias']
        self.truncate = float(truncate)
        self.max_voxels = max_voxels

    def inspect(self, pair: ScanPair) -> dict:
        """Header-only preflight, including resource rejection BEFORE voxel reads."""
        ih, lh = read_header(pair.image_path), read_header(pair.label_path)
        if not compare_geometry(ih, lh)['consistent']:
            raise ValueError('image/label geometry mismatch')
        for header in (ih, lh):
            if header['spatial_unit_code'] != 2 or not header['axis_aligned']:
                raise ValueError('full-scan v1 requires cardinal grids with millimetre units')
            if 'nonfinite_scaling_assumed_identity' in header['notices']:
                raise ValueError('nonfinite scaling metadata is not accepted for model input')
        shape, target, grid = build_target_grid(ih, self.spacing)
        if self.max_voxels is not None and max(ih['voxel_count'], math.prod(shape)) > self.max_voxels:
            raise ValueError('CPU synthetic voxel limit exceeded; refusing to shrink/crop the scan')
        # (d,h,w) -> (x,y,z) = (w,h,d); includes orientation and voxel centres.
        permutation = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
        tensor_affine = target @ permutation
        metadata = {
            'case_id': pair.case_id, 'candidate': self.candidate,
            'spacing_ras_mm': list(self.spacing), 'shape_dhw': list(shape[::-1]),
            'tensor_axes': ['S', 'A', 'R'], 'grid': grid,
            'tensor_dhw_to_ras_mm': tensor_affine.tolist(),
            'ct_interpolation': 'trilinear', 'label_interpolation': 'nearest',
            'boundary': 'nearest_edge_replication', 'antialias': self.antialias,
            'gaussian_truncate': self.truncate, 'intensity': 'scaled_hu',
            'batch_size': 1, 'crop': False, 'network_padding': False,
        }
        for role, header in (('image', ih), ('label', lh)):
            original = affine4(header)
            metadata[role] = {
                'original_shape_ijk': header['shape_native'],
                'original_affine_ras_mm': original.tolist(),
                'scaling_effective': header['scaling_effective'],
                'header_sha256': header['header_sha256'],
                'file_size_bytes': header['file_size_bytes'], 'mtime_ns': header['mtime_ns'],
                'notices': header['notices'],
                'tensor_dhw_to_original_ijk': np.linalg.solve(original, tensor_affine).tolist(),
                'original_ijk_to_tensor_dhw': np.linalg.solve(tensor_affine, original).tolist(),
            }
        return metadata

    def __call__(self, pair: ScanPair) -> FullScanSample:
        metadata = self.inspect(pair)
        ct, label, ih, lh, checks = load_pair(pair.image_path, pair.label_path)
        for role, header in (('image', ih), ('label', lh)):
            if header['header_sha256'] != metadata[role]['header_sha256']:
                raise ValueError('header changed after preflight')
        shape = tuple(metadata['grid']['shape_xyz'])
        target = np.array(metadata['grid']['affine_ras_mm'])
        image = resample(ct, affine4(ih), shape, target, 1, self.antialias, self.truncate)
        del ct
        labels = resample(label, affine4(lh), shape, target, 0)
        del label
        if not np.isfinite(image).all() or labels.min() < 0 or labels.max() > 15:
            raise ValueError('resampling produced nonfinite image or illegal labels')
        # Contiguous positive-stride arrays, including flipped original orientations.
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 1, 0))).unsqueeze(0).unsqueeze(0)
        labels = torch.from_numpy(np.ascontiguousarray(labels.transpose(2, 1, 0), dtype=np.int64)).unsqueeze(0)
        metadata['antialias_sigma_native_voxels'] = (
            antialias_sigmas(affine4(ih), target).tolist() if self.antialias else [0., 0., 0.])
        metadata['checks'] = checks
        for role, path in (('image', pair.image_path), ('label', pair.label_path)):
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (metadata[role]['file_size_bytes'], metadata[role]['mtime_ns']):
                raise ValueError('source file changed during preprocessing')
        return FullScanSample(image, labels, metadata)


class FullScanDataset(Dataset):
    """Lazy CPU Dataset of complete cases, each ALREADY batch size one.

    Use dataset[index] directly for the single-case probe. Do not apply default
    DataLoader batching: it would add a second batch axis. No volume cache.
    """

    def __init__(self, pairs: list[ScanPair], preprocessor: FullScanPreprocessor):
        if not pairs or len({pair.case_id for pair in pairs}) != len(pairs):
            raise ValueError('case list must be nonempty with unique identifiers')
        self.pairs = tuple(pairs)
        self.preprocessor = preprocessor

    @classmethod
    def from_amos_training(cls, data_root: Path, case_ids: list[str], selection_config: dict,
                           preprocessor: FullScanPreprocessor):
        records, _, _ = select_training(data_root, selection_config)
        lookup = {record['case_id']: record for record in records}
        if any(case_id not in lookup for case_id in case_ids):
            raise ValueError('requested case must belong to the configured AMOS training CT split')
        return cls([ScanPair(case_id, contained_path(data_root, lookup[case_id]['image']),
                             contained_path(data_root, lookup[case_id]['label'])) for case_id in case_ids], preprocessor)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int) -> FullScanSample:
        return self.preprocessor(self.pairs[index])
