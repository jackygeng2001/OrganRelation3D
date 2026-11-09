"""Light BraTS training augmentation: transform, determinism, resume and dev isolation."""
import unittest
from unittest.mock import patch

import torch

import support as fixtures
from test_brain_tumor import Cases, config
from organ_relation.data import augmentation as augment
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.state import load_checkpoint, seed_all

SPEC = dict(kind='flip_intensity_v1', flip_probability=0.5, intensity_scale=0.1, intensity_shift=0.1)


class TransformTests(unittest.TestCase):
    def volume(self):
        g = torch.Generator().manual_seed(3)
        image = torch.randn(1, 4, 5, 6, 7, generator=g)
        image[:, :, 0] = 0  # Outside-brain voxels are exactly zero after z-scoring.
        image[:, 2, :, :, :2] = 0
        label = (image[:, 0] > 0.3).long() + (image[:, 1] > 0.8).long()
        return image, label

    def test_specification_is_validated(self):
        augment.validate_augmentation(None)
        augment.validate_augmentation(SPEC)
        for bad in (dict(SPEC, kind='other'), dict(SPEC, extra=1), {k: v for k, v in SPEC.items() if k != 'intensity_shift'},
                    dict(SPEC, flip_probability=1.5), dict(SPEC, intensity_scale=1.0), dict(SPEC, intensity_shift=-0.1),
                    dict(SPEC, flip_probability=True), dict(SPEC, intensity_scale=float('nan'))):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                augment.validate_augmentation(bad)

    def test_draws_depend_only_on_seed_and_step_and_stay_in_range(self):
        first = augment.draw(SPEC, augment.step_generator(7, 11), 4)
        self.assertEqual(first, augment.draw(SPEC, augment.step_generator(7, 11), 4))
        draws = [augment.draw(SPEC, augment.step_generator(7, step), 4) for step in range(1, 401)]
        self.assertGreater(len({str(d) for d in draws}), 390)
        flips = torch.tensor([d['flip'] for d in draws], dtype=torch.float64)
        self.assertTrue(((flips.mean(0) > 0.42) & (flips.mean(0) < 0.58)).all())
        for d in draws:
            self.assertTrue(all(0.9 <= s <= 1.1 for s in d['scale']) and all(-0.1 <= o <= 0.1 for o in d['shift']))

    def test_identity_parameters_change_nothing(self):
        image, label = self.volume()
        out, out_label = augment.apply(image, label, dict(flip=[False] * 3, scale=[1.0] * 4, shift=[0.0] * 4))
        self.assertTrue(torch.equal(out, image) and torch.equal(out_label, label))

    def test_intensity_only_on_nonzero_voxels_per_modality(self):
        image, label = self.volume()
        before = image.clone()
        params = dict(flip=[False] * 3, scale=[0.9, 1.0, 1.1, 1.05], shift=[0.1, -0.05, 0.0, 0.02])
        out, out_label = augment.apply(image, label, params)
        self.assertTrue(torch.equal(image, before))  # Inputs are not modified in place.
        self.assertTrue(torch.equal(out == 0, image == 0))
        for c in range(4):
            mask = image[:, c] != 0
            torch.testing.assert_close(out[:, c][mask], image[:, c][mask] * params['scale'][c] + params['shift'][c])
        self.assertTrue(torch.equal(out_label, label))

    def test_flips_move_image_and_label_together(self):
        image, label = self.volume()
        for flip in ([True, False, False], [False, True, True], [True, True, True]):
            with self.subTest(flip=flip):
                out, out_label = augment.apply(image, label, dict(flip=flip, scale=[1.0] * 4, shift=[0.0] * 4))
                dims = [2 + a for a, f in enumerate(flip) if f]
                self.assertTrue(torch.equal(out, image.flip(dims)))
                self.assertTrue(torch.equal(out_label, (out[:, 0] > 0.3).long() + (out[:, 1] > 0.8).long()))


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def trainer(self, path, *, resume=False, augmentation=SPEC, mode='brats_regions', validation=False, steps=4):
        seed_all(171)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']))
        options = dict(self.cfg['training'], max_steps=steps, validation_every=1 if validation else 0,
                       diagnostics_every=1, checkpoint_every=1, augmentation=augmentation,
                       validation_role='train_monitor')
        identity = dict(mode=mode, model=self.cfg['model'], loss=self.cfg['loss'], threshold=.5, training=options,
                        preprocessing=self.cfg['preprocessing'],
                        provenance='test', data={'manifest_hash': 'synthetic'})
        extra = dict(validation_dataset=Cases(), validation_case_ids=['one', 'two']) if validation else {}
        return Trainer(model, RegionJointLoss(**self.cfg['loss']),
                       torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False),
                       Cases(), ['one', 'two'], device='cpu', options=options, identity=identity, run_dir=path,
                       resume=path / 'last.ckpt' if resume else None, tensorboard=False, **extra)

    def test_options_reject_invalid_specs_and_other_modes(self):
        with self.assertRaises(ValueError):
            validate_options(dict(self.cfg['training'], max_steps=1, augmentation=dict(SPEC, flip_probability=2)))
        with self.assertRaisesRegex(ValueError, 'only for BraTS'):
            self.trainer(self.fixture.root / 'other', mode='organ_relation_joint')

    def test_resume_reproduces_every_augmented_step(self):
        whole = self.trainer(self.fixture.root / 'whole'); self.fixture.quiet_run(whole)
        part = self.trainer(self.fixture.root / 'part'); self.fixture.quiet_run(part, stop_after=2)
        resumed = self.trainer(part.run_dir, resume=True); self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(part.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator'):
            self.fixture.assert_nested_equal(a[key], b[key])
        rows = lambda run: [r['augmentation'] for r in self.fixture.rows(run) if r['phase'] == 'train']
        self.assertEqual(rows(whole.run_dir), rows(part.run_dir))
        self.assertEqual(len(rows(whole.run_dir)), 4)
        expected = [augment.draw(SPEC, augment.step_generator(self.cfg['training']['seed'], step), 4) for step in range(1, 5)]
        self.assertEqual(rows(whole.run_dir), expected)

    def test_only_training_cases_are_augmented(self):
        calls = []
        original = augment.apply
        def counted(image, label, params):
            calls.append(params)
            return original(image, label, params)
        trainer = self.trainer(self.fixture.root / 'dev', validation=True, steps=2)
        with patch.object(augment, 'apply', side_effect=counted):
            self.fixture.quiet_run(trainer)
        self.assertEqual(len(calls), 2)  # Two training steps; four dev forwards were not augmented.
        self.assertEqual(len([r for r in self.fixture.rows(trainer.run_dir) if r['phase'] == 'train_monitor']), 2)

    def test_disabled_augmentation_keeps_rows_unchanged(self):
        trainer = self.trainer(self.fixture.root / 'plain', augmentation=None, steps=2)
        self.fixture.quiet_run(trainer)
        self.assertTrue(all('augmentation' not in r for r in self.fixture.rows(trainer.run_dir)))


if __name__ == '__main__':
    unittest.main()
