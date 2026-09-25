"""Synthetic NIfTI-to-Tensor contracts, with real NumPy/PyTorch bridging."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import nibabel as nib
import numpy as np
import torch
from organ_relation.data.full_scan import FullScanDataset, FullScanPreprocessor, ScanPair


def configuration():
    return json.loads((ROOT / 'configs/baseline.json').read_text(encoding='utf-8'))['preprocessing']


def synthetic_pair(root, *, shape=(8, 6, 4), slope=2., intercept=-1024., label_scale=1.):
    for folder in ('imagesTr', 'labelsTr'):
        (root / folder).mkdir(exist_ok=True)
    affine = np.diag([-1.5, 1.5, 3., 1.]); affine[:3, 3] = [12., -8., 30.]
    raw = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    labels = (raw % (8 if label_scale == 2 else 16)).astype(np.uint8)
    pair = ScanPair('amos_0001', root / 'imagesTr/amos_0001.nii.gz', root / 'labelsTr/amos_0001.nii.gz')
    for path, values, scaling in ((pair.image_path, raw, (slope, intercept)), (pair.label_path, labels, (label_scale, 0.))):
        image = nib.Nifti1Image(values, affine)
        image.header.set_xyzt_units('mm')
        image.header.set_slope_inter(*scaling)
        nib.save(image, path)
    (root / 'dataset.json').write_text(json.dumps({'training': [{'image': 'imagesTr/amos_0001.nii.gz', 'label': 'labelsTr/amos_0001.nii.gz'}]}), encoding='utf-8')
    return pair, raw, labels


class FullScanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pair, self.raw, self.labels = synthetic_pair(self.root)

    def test_analytic_orientation_scaling_alignment_shapes_and_dtypes(self):
        sample = FullScanPreprocessor(configuration(), 'A')(self.pair)
        expected_image = (self.raw.astype(np.float32)*2-1024)[::-1].transpose(2, 1, 0)
        expected_label = self.labels[::-1].transpose(2, 1, 0)
        np.testing.assert_array_equal(sample.image[0, 0].numpy(), expected_image)
        np.testing.assert_array_equal(sample.label[0].numpy(), expected_label)
        self.assertEqual(sample.image.shape, (1, 1, 4, 6, 8))
        self.assertEqual(sample.label.shape, (1, 4, 6, 8))
        self.assertEqual(sample.image.dtype, torch.float32)
        self.assertEqual(sample.label.dtype, torch.int64)
        self.assertTrue(sample.image.is_contiguous() and sample.label.is_contiguous())
        self.assertFalse(sample.image.requires_grad or sample.label.requires_grad)

    def test_all_candidates_preserve_boundaries_center_and_valid_classes(self):
        for candidate, shape in [('A', (4, 6, 8)), ('B', (4, 5, 6)), ('C', (3, 5, 6))]:
            with self.subTest(candidate=candidate):
                sample = FullScanPreprocessor(configuration(), candidate)(self.pair)
                self.assertEqual(sample.image.shape[2:], shape)
                self.assertEqual(sample.label.shape[1:], shape)
                self.assertTrue(set(sample.label.unique().tolist()).issubset(set(self.labels.ravel())))
                grid = sample.metadata['grid']
                old_lo, old_hi = np.array(grid['original_boundary_min_ras_mm']), np.array(grid['original_boundary_max_ras_mm'])
                new_lo, new_hi = np.array(grid['target_boundary_min_ras_mm']), np.array(grid['target_boundary_max_ras_mm'])
                self.assertTrue(np.all(new_lo <= old_lo) and np.all(new_hi >= old_hi))
                np.testing.assert_allclose((old_lo+old_hi)/2, (new_lo+new_hi)/2)
                self.assertFalse(sample.metadata['crop'] or sample.metadata['network_padding'])

    def test_tensor_world_mapping_and_inverse_for_image_and_label(self):
        sample = FullScanPreprocessor(configuration(), 'B')(self.pair)
        tensor_affine = np.array(sample.metadata['tensor_dhw_to_ras_mm'])
        for role in ('image', 'label'):
            meta = sample.metadata[role]
            forward = np.array(meta['tensor_dhw_to_original_ijk'])
            inverse = np.array(meta['original_ijk_to_tensor_dhw'])
            np.testing.assert_allclose(forward @ inverse, np.eye(4), atol=1e-12)
            np.testing.assert_allclose(np.array(meta['original_affine_ras_mm']) @ forward, tensor_affine, atol=1e-12)
        self.assertEqual(sample.metadata['tensor_axes'], ['S', 'A', 'R'])

    def test_label_scaling_and_ct_zero_intercept_are_independent(self):
        pair, raw, labels = synthetic_pair(self.root, slope=1., intercept=0., label_scale=2.)
        sample = FullScanPreprocessor(configuration(), 'A')(pair)
        np.testing.assert_array_equal(sample.image[0, 0].numpy(), raw[::-1].transpose(2, 1, 0))
        np.testing.assert_array_equal(sample.label[0].numpy(), (labels*2)[::-1].transpose(2, 1, 0))

    def test_gt_changes_do_not_change_image_or_target_grid(self):
        processor = FullScanPreprocessor(configuration(), 'B')
        first = processor(self.pair)
        original = nib.load(self.pair.label_path)
        replacement = nib.Nifti1Image(np.zeros(self.labels.shape, np.uint8), original.affine, original.header)
        nib.save(replacement, self.pair.label_path)
        second = processor(self.pair)
        self.assertTrue(torch.equal(first.image, second.image))
        self.assertEqual(first.metadata['grid'], second.metadata['grid'])
        self.assertEqual(second.label.count_nonzero(), 0)

    def test_misalignment_rejected_before_voxel_read(self):
        original = nib.load(self.pair.label_path)
        affine = original.affine.copy(); affine[0, 3] += 1.
        nib.save(nib.Nifti1Image(self.labels, affine, original.header), self.pair.label_path)
        with patch('organ_relation.data.full_scan.load_pair', side_effect=AssertionError('voxel read')):
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                FullScanPreprocessor(configuration(), 'A')(self.pair)

    def test_illegal_scaled_label_is_rejected(self):
        pair, _, _ = synthetic_pair(self.root, label_scale=.5)
        with self.assertRaisesRegex(ValueError, 'integer labels'):
            FullScanPreprocessor(configuration(), 'A')(pair)

    def test_resource_guard_runs_before_loading_and_never_reduces_shape(self):
        with patch('organ_relation.data.full_scan.load_pair', side_effect=AssertionError('voxel read')):
            with self.assertRaisesRegex(ValueError, 'voxel limit'):
                FullScanPreprocessor(configuration(), 'A', max_voxels=10)(self.pair)

    def test_source_files_unchanged(self):
        before = {p: p.read_bytes() for p in (self.pair.image_path, self.pair.label_path)}
        FullScanPreprocessor(configuration(), 'C')(self.pair)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_lazy_dataset_single_batched_sample_and_training_selection(self):
        processor = FullScanPreprocessor(configuration(), 'A')
        selection = json.loads((ROOT / 'configs/ct_stats.json').read_text())
        with patch('organ_relation.data.full_scan.load_pair', side_effect=AssertionError('eager read')):
            dataset = FullScanDataset.from_amos_training(self.root, ['amos_0001'], selection, processor)
            self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].image.shape[0], 1)
        with self.assertRaises(ValueError):
            FullScanDataset.from_amos_training(self.root, ['amos_0500'], selection, processor)
        with self.assertRaises(ValueError):
            FullScanDataset([self.pair, self.pair], processor)

    def test_non_mm_and_oblique_inputs_are_explicitly_rejected(self):
        for unit in ('meter', 'unknown'):
            original = nib.load(self.pair.image_path)
            original.header.set_xyzt_units(unit)
            nib.save(original, self.pair.image_path)
            with self.assertRaises(ValueError):
                FullScanPreprocessor(configuration(), 'A')(self.pair)
            synthetic_pair(self.root)
        for path in (self.pair.image_path, self.pair.label_path):
            original = nib.load(path); affine = original.affine.copy(); affine[0, 1] = .1
            nib.save(nib.Nifti1Image(np.asanyarray(original.dataobj), affine, original.header), path)
        with self.assertRaisesRegex(ValueError, 'cardinal'):
            FullScanPreprocessor(configuration(), 'A')(self.pair)

    def test_singleton_axes_and_invalid_configuration(self):
        pair, _, _ = synthetic_pair(self.root, shape=(1, 3, 1))
        sample = FullScanPreprocessor(configuration(), 'A')(pair)
        self.assertEqual(sample.image.shape, (1, 1, 1, 3, 1))
        for key, value in [('grid', 'crop'), ('intensity', 'unknown'), ('boundary', 'zero'), ('antialias', 1), ('gaussian_truncate', 0)]:
            config = copy.deepcopy(configuration()); config[key] = value
            with self.assertRaises(ValueError):
                FullScanPreprocessor(config, 'A')


if __name__ == '__main__':
    unittest.main()
