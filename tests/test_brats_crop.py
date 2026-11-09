"""Image-only nonzero brain-box crop: geometry, data equivalence and GT guard."""
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

import support as fixtures
from organ_relation.data.brats import (BraTSDataset, MODALITIES, expand_box, index_cases,
                                       nonzero_bbox, validate_spatial_crop)
from organ_relation.training.brain_tumor import region_dice

CROP = dict(kind='nonzero_bbox', margin=0, multiple_of=1)


def preprocessing():
    return json.loads((fixtures.ROOT / 'configs/development/implementation_check.json').read_text())['preprocessing']


class BoxTests(unittest.TestCase):
    def test_tight_box_margin_and_rounding_always_contain_support(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            shape = tuple(int(n) for n in rng.integers(1, 30, 3))
            support = np.zeros(shape, bool)
            low = [int(rng.integers(0, n)) for n in shape]
            high = [int(rng.integers(a, n)) + 1 for a, n in zip(low, shape)]
            support[tuple(slice(a, b) for a, b in zip(low, high))] = rng.random(
                [b - a for a, b in zip(low, high)]) < .3
            support[tuple(low)] = True
            index = np.argwhere(support)
            margin, multiple = int(rng.integers(0, 4)), int(rng.integers(1, 17))
            start, stop = nonzero_bbox(support, margin=margin, multiple_of=multiple)
            for axis, size in enumerate(shape):
                self.assertLessEqual(0, start[axis])
                self.assertLessEqual(stop[axis], size)
                self.assertLessEqual(start[axis], index[:, axis].min())
                self.assertGreater(stop[axis], index[:, axis].max())
                length = stop[axis] - start[axis]
                self.assertTrue(length % multiple == 0 or length == size)
            cropped = support[tuple(slice(a, b) for a, b in zip(start, stop))]
            self.assertEqual(cropped.sum(), support.sum())

    def test_tight_box_is_exact_and_rounding_is_centred_then_shifted_inside(self):
        support = np.zeros((10, 20, 30), bool)
        support[2:5, 3:9, 0:7] = True
        self.assertEqual(nonzero_bbox(support, margin=0, multiple_of=1), ([2, 3, 0], [5, 9, 7]))
        self.assertEqual(expand_box([(2, 5), (3, 9), (0, 7)], (10, 20, 30), margin=1, multiple_of=8),
                         ([0, 2, 0], [8, 10, 8]))
        self.assertEqual(expand_box([(8, 10), (3, 9), (0, 30)], (10, 20, 30), margin=0, multiple_of=16),
                         ([0, 0, 0], [10, 16, 30]))
        with self.assertRaises(ValueError):
            nonzero_bbox(np.zeros((2, 2, 2), bool), margin=0, multiple_of=1)

    def test_crop_config_is_explicit(self):
        self.assertIsNone(validate_spatial_crop(None))
        self.assertEqual(validate_spatial_crop(CROP), CROP)
        for bad in (dict(kind='patch', margin=0, multiple_of=1), dict(CROP, margin=-1),
                    dict(CROP, multiple_of=0), dict(CROP, margin=True), dict(kind='nonzero_bbox'),
                    dict(CROP, extra=1)):
            with self.assertRaises(ValueError):
                validate_spatial_crop(bad)


class CropDatasetTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.root, ignore_errors=True))

    def make_source(self, *, tumor_outside=False):
        """XYZ (12,10,8) volumes; brain occupies x2:9, y1:7, z3:8 (modality 3 adds z2)."""
        source = self.root / 'data'
        source.mkdir()
        rng = np.random.default_rng(1)
        case = 'BraTS-GLI-00001-000'
        folder = source / case
        folder.mkdir()
        for index, role in enumerate((*MODALITIES, 'seg')):
            array = np.zeros((12, 10, 8), np.float32)
            if role == 'seg':
                array[4:7, 2:5, 4:6] = 2
                array[5, 3, 5] = 1
                array[5, 4, 5] = 4 - 1  # canonical 2023 encoding uses 3 for ET
                if tumor_outside:
                    array[11, 9, 0] = 2
                array = array.astype(np.int16)
            else:
                array[2:9, 1:7, 3:8] = rng.uniform(1, 100, (7, 6, 5)) * (index + 1)
                if index == 3:
                    array[5, 5, 2] = 7.0  # FLAIR-only voxel extends the union box
            nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f'{case}-{role}.nii.gz')
        return index_cases(source, cohort='brats2023_gli')

    def test_crop_equals_slice_of_full_volume_with_identical_zscore(self):
        manifest = self.make_source()
        full = BraTSDataset(manifest, **preprocessing())[0]
        cropped = BraTSDataset(manifest, **preprocessing(), spatial_crop=CROP)[0]
        info = cropped.metadata['spatial_crop']
        # DHW = reversed XYZ; union of all four modalities, z from 2 (FLAIR) to 8.
        self.assertEqual(info['start_dhw'], [2, 1, 2])
        self.assertEqual(info['stop_dhw'], [8, 7, 9])
        self.assertEqual(info['original_shape_dhw'], [8, 10, 12])
        self.assertEqual(cropped.metadata['shape_dhw'], [6, 6, 7])
        window = tuple(slice(a, b) for a, b in zip(info['start_dhw'], info['stop_dhw']))
        torch.testing.assert_close(cropped.image, full.image[(slice(None), *window)], rtol=0, atol=0)
        self.assertTrue(torch.equal(cropped.label, full.label[window]))
        self.assertEqual(cropped.metadata['normalization'], full.metadata['normalization'])
        self.assertTrue(cropped.image.is_contiguous() and cropped.label.is_contiguous())
        self.assertEqual(cropped.label.dtype, torch.int64)
        self.assertNotIn('spatial_crop', full.metadata)
        self.assertEqual(full.metadata['shape_dhw'], [8, 10, 12])

    def test_rounded_crop_keeps_every_nonzero_voxel(self):
        manifest = self.make_source()
        full = BraTSDataset(manifest, **preprocessing())[0]
        cropped = BraTSDataset(manifest, **preprocessing(),
                               spatial_crop=dict(kind='nonzero_bbox', margin=1, multiple_of=4))[0]
        self.assertTrue(all(n % 4 == 0 or n == o for n, o in
                            zip(cropped.metadata['shape_dhw'], [8, 10, 12])))
        self.assertEqual(int((cropped.image != 0).sum()), int((full.image != 0).sum()))
        self.assertEqual(int((cropped.label > 0).sum()), int((full.label > 0).sum()))

    def test_crop_refuses_to_drop_labeled_tumor(self):
        manifest = self.make_source(tumor_outside=True)
        BraTSDataset(manifest, **preprocessing())[0]  # Full grid still loads.
        with self.assertRaisesRegex(ValueError, 'discard labeled tumor'):
            BraTSDataset(manifest, **preprocessing(), spatial_crop=CROP)[0]

    def test_cropped_dice_equals_full_grid_dice_with_background_outside(self):
        manifest = self.make_source()
        sample = BraTSDataset(manifest, **preprocessing(), spatial_crop=CROP)[0]
        full = BraTSDataset(manifest, **preprocessing())[0]
        info = sample.metadata['spatial_crop']
        logits = torch.randn(1, 3, *sample.label.shape, generator=torch.Generator().manual_seed(2))
        pad = [p for a, b, n in reversed(list(zip(info['start_dhw'], info['stop_dhw'],
                                                    info['original_shape_dhw']))) for p in (a, n - b)]
        pasted = F.pad(logits, pad, value=-30.0)  # Outside the box is predicted background.
        self.assertEqual(region_dice(logits, sample.label[None], .5),
                         region_dice(pasted, full.label[None], .5))


if __name__ == '__main__':
    unittest.main()
