"""Exact .npy loader cache: bitwise equality, no source reads on hit, keyed invalidation."""
import contextlib
import importlib.util
import io
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import test_training as fixtures
from organ_relation.data.brats import BraTSDataset, MODALITIES, index_cases

CROP = dict(kind='nonzero_bbox', margin=0, multiple_of=4)


def preprocessing(**extra):
    base = json.loads((fixtures.ROOT / 'configs/brats_regions.json').read_text())['preprocessing']
    return dict(base, **extra)


def write_source(root):
    source = root / 'data'
    rng = np.random.default_rng(3)
    for case in ('BraTS-GLI-00001-000', 'BraTS-GLI-00002-000'):
        folder = source / case
        folder.mkdir(parents=True)
        for index, role in enumerate((*MODALITIES, 'seg')):
            array = np.zeros((12, 10, 8), np.float32)
            if role == 'seg':
                array[4:7, 2:5, 4:6] = 2
                array[5, 3, 5] = 3
                array = array.astype(np.int16)
            else:
                array[2:9, 1:7, 3:8] = rng.uniform(1, 100, (7, 6, 5)) * (index + 1)
            nib.save(nib.Nifti1Image(array, np.eye(4)), folder / f'{case}-{role}.nii.gz')
    return source


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        self.source = write_source(self.root)
        self.cache = self.root / 'cache'

    def manifest(self):
        return index_cases(self.source, cohort='brats2023_gli')

    def assert_same(self, a, b):
        self.assertTrue(torch.equal(a.image, b.image))
        self.assertEqual(a.image.dtype, b.image.dtype)
        self.assertTrue(torch.equal(a.label, b.label))
        self.assertEqual(a.label.dtype, torch.int64)
        self.assertEqual(a.metadata, b.metadata)
        self.assertTrue(a.image.is_contiguous() and a.label.is_contiguous())

    def entries(self):
        return sorted(self.cache.rglob('*.json'))

    def test_cache_is_bitwise_identical_and_hit_never_reads_source(self):
        for extra in (dict(), dict(spatial_crop=CROP)):
            with self.subTest(extra=extra):
                plain = BraTSDataset(self.manifest(), **preprocessing(**extra))
                expected = [plain[index] for index in range(2)]
                cached = BraTSDataset(self.manifest(), **preprocessing(**extra), cache_dir=self.cache)
                for index in range(2):
                    self.assert_same(cached[index], expected[index])  # Miss: build + write.
                with patch.object(BraTSDataset, '_read', side_effect=AssertionError('source read on hit')):
                    for index in range(2):
                        self.assert_same(cached[index], expected[index])  # Hit.
        self.assertEqual(len(self.entries()), 4)  # Two preprocessing settings x two cases.
        self.assertEqual(list(self.cache.rglob('*.tmp')), [])
        label = np.load(next(self.cache.rglob('*.label.npy')))
        self.assertEqual(label.dtype, np.uint8)

    def test_source_change_creates_new_entry_instead_of_reuse(self):
        BraTSDataset(self.manifest(), **preprocessing(), cache_dir=self.cache)[0]
        path = self.source / 'BraTS-GLI-00001-000' / 'BraTS-GLI-00001-000-t2f.nii.gz'
        array = np.asanyarray(nib.load(path).dataobj).copy()
        array[3, 3, 4] += 1000
        time.sleep(0.01)
        nib.save(nib.Nifti1Image(array, np.eye(4)), path)
        fresh = BraTSDataset(self.manifest(), **preprocessing(), cache_dir=self.cache)[0]
        plain = BraTSDataset(self.manifest(), **preprocessing())[0]
        self.assert_same(fresh, plain)
        self.assertEqual(len(self.entries()), 2)

    def test_corrupt_or_tampered_entry_is_refused(self):
        dataset = BraTSDataset(self.manifest(), **preprocessing(), cache_dir=self.cache)
        dataset[0]
        image = next(self.cache.rglob('*.image.npy'))
        image.write_bytes(image.read_bytes()[:100])
        with self.assertRaisesRegex(ValueError, 'unreadable data cache entry'):
            dataset[0]
        shutil.rmtree(self.cache)
        dataset[0]
        entry = self.entries()[0]
        content = json.loads(entry.read_text())
        content['image_shape'][0] += 1
        entry.write_text(json.dumps(content))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            dataset[0]

    def test_cache_inside_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside the read-only source'):
            BraTSDataset(self.manifest(), **preprocessing(), cache_dir=self.source / 'cache')

    def test_cli_check_reads_through_cache(self):
        spec = importlib.util.spec_from_file_location('brats_cli', fixtures.ROOT / 'scripts/brats.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = ['check', '--device', 'cpu', '--cohort', 'brats2023_gli', '--source', str(self.source),
                '--case', 'BraTS-GLI-00001-000', '--data-cache', str(self.cache)]
        for _ in range(2):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(cli.main(args), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'passed')
        self.assertEqual(len(self.entries()), 1)


if __name__ == '__main__':
    unittest.main()
