"""Single-variable balanced-CE formula, identity and CPU integration tests."""
import contextlib
import importlib.util
import io
import json
import shutil
import unittest

import torch
import test_backbone_only as backbone
import test_training as fixtures
from organ_relation.losses import JointLoss, SegmentationLoss
from organ_relation.training.engine import final_diagnostic_metrics
from organ_relation.metrics import hard_dice
from organ_relation.training.state import atomic_json, load_checkpoint

BALANCED = 'foreground_background_balanced'
ROOT = fixtures.ROOT


class BalancedCETests(unittest.TestCase):
    def setUp(self):
        self.fixture = backbone.BackboneOnlyTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        torch.manual_seed(818)

    def example(self, dtype=torch.float64):
        logits = torch.randn(2, 16, 1, 2, 4, dtype=dtype, requires_grad=True)
        # Unequal FG fractions and unequal class sizes distinguish case/group
        # averaging from pooled batch or 15-class equal-weight CE.
        labels = torch.tensor([[[[0, 0, 0, 0], [0, 0, 0, 1]]],
                               [[[0, 1, 1, 1], [2, 2, 3, 15]]]])
        return logits, labels

    def test_voxel_mean_default_explicit_and_historical_ce_bit_exact(self):
        for dtype in (torch.float32, torch.float64):
            logits, label = self.example(dtype)
            default = SegmentationLoss(epsilon=1e-6)(logits, label)
            explicit = SegmentationLoss(epsilon=1e-6, ce_reduction_mode='voxel_mean')(logits, label)
            historical = -torch.log(logits.softmax(1).flatten(2).gather(
                1, label.flatten(1).unsqueeze(1)).squeeze(1) + 1e-6).mean(1)
            self.assertTrue(torch.equal(default.final.ce, historical))
            self.assertTrue(torch.equal(default.total, explicit.total))
            for a, b in zip(default.final, explicit.final):
                self.assertTrue(torch.equal(a, b))
            coarse = torch.randn(2, 16, 1, 1, 2, dtype=dtype)
            joint = JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False)(coarse, logits, label)
            for a, b in zip(default.final, joint.final):
                self.assertTrue(torch.equal(a, b))
            ga = torch.autograd.grad(default.total, logits, retain_graph=True)[0]
            gb = torch.autograd.grad(explicit.total, logits)[0]
            self.assertTrue(torch.equal(ga, gb))

    def test_balanced_formula_independent_case_reference_and_unchanged_dice(self):
        logits, label = self.example()
        actual = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)(logits, label)
        reference = []
        for b in range(2):
            p = logits[b].softmax(0).flatten(1)
            y = label[b].flatten()
            terms = torch.stack([-torch.log(p[int(c), x] + 1e-6) for x, c in enumerate(y)])
            reference.append(.5 * terms[y == 0].mean() + .5 * terms[y > 0].mean())
        reference = torch.stack(reference)
        torch.testing.assert_close(actual.final.ce, reference, rtol=1e-14, atol=1e-14)
        original = SegmentationLoss(epsilon=1e-6)(logits, label)
        self.assertTrue(torch.equal(actual.final.dice_per_class, original.final.dice_per_class))
        self.assertTrue(torch.equal(actual.final.dice_loss, original.final.dice_loss))
        torch.testing.assert_close(actual.total, (reference + original.final.dice_loss).mean())
        grads = torch.autograd.grad(actual.total, logits, retain_graph=True)[0]
        expected = torch.autograd.grad((reference + original.final.dice_loss).mean(), logits)[0]
        torch.testing.assert_close(grads, expected, rtol=1e-12, atol=1e-12)

    def test_per_case_batch_mean_differs_from_pooled_batch(self):
        logits, label = self.example()
        criterion = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)
        batched = criterion(logits, label)
        singles = [criterion(logits[i:i+1], label[i:i+1]).total for i in range(2)]
        self.assertTrue(torch.equal(batched.total, torch.stack(singles).mean()))
        losses = -torch.log(logits.softmax(1).gather(1, label.unsqueeze(1)).squeeze(1) + 1e-6)
        wrong = .5 * losses[label == 0].mean() + .5 * losses[label > 0].mean()
        self.assertGreater(abs(wrong.item() - batched.final.ce.mean().item()), 1e-3)

    def test_fp64_balanced_gradcheck(self):
        logits = torch.randn(1, 16, 1, 1, 3, dtype=torch.float64, requires_grad=True)
        label = torch.tensor([[[[0, 0, 11]]]])
        criterion = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)
        self.assertTrue(torch.autograd.gradcheck(lambda x: criterion(x, label).total, (logits,)))

    def test_empty_group_rejected_without_changing_default_empty_case_behavior(self):
        logits = torch.randn(2, 16, 1, 2, 2)
        balanced = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)
        for value in (0, 1):
            label = torch.full((2, 1, 2, 2), value)
            with self.assertRaisesRegex(ValueError, 'both background and foreground'):
                balanced(logits, label)
            self.assertTrue(torch.isfinite(SegmentationLoss(epsilon=1e-6)(logits, label).total))
        with self.assertRaisesRegex(ValueError, 'ce_reduction_mode'):
            SegmentationLoss(epsilon=1e-6, ce_reduction_mode='class_balanced')

    def test_config_exactly_one_added_field(self):
        old = json.loads((ROOT / 'configs/train_backbone_only_overfit.json').read_text())
        new = json.loads((ROOT / 'configs/train_backbone_only_overfit_balanced_ce.json').read_text())
        self.assertEqual(new.pop('ce_reduction_mode'), BALANCED)
        self.assertEqual(old, new)

    def test_monitor_ce_means_use_same_formula_and_retain_existing_metrics(self):
        logits, label = self.example()
        logits, label = logits[:1], label[:1]
        criterion = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)
        record = final_diagnostic_metrics(logits, label, criterion, hard_dice(logits.argmax(1)[0], label[0]))
        losses = -torch.log(logits.softmax(1).gather(1, label.unsqueeze(1)).squeeze(1) + 1e-6)
        self.assertAlmostEqual(record['CE_bg_mean'], losses[label == 0].mean().item())
        self.assertAlmostEqual(record['CE_fg_mean'], losses[label > 0].mean().item())
        self.assertEqual(record['balanced_ce'], criterion(logits, label).final.ce.item())
        self.assertEqual(record['ce_reduction_mode'], BALANCED)
        for key in ('final_soft_dice', 'predicted_foreground_voxels', 'foreground_true_positive_voxels',
                    'gt_foreground_true_class_mean_probability', 'gt_foreground_background_mean_probability'):
            self.assertIn(key, record)

    def test_balanced_resume_exact_and_voxel_checkpoint_rejected(self):
        def trainer(path, **kwargs):
            return self.fixture.trainer(path, ce_reduction_mode=BALANCED, **kwargs)
        whole = trainer(self.root / 'whole'); whole.run()
        first = trainer(self.root / 'split'); first.run(stop_after=2)
        resumed = trainer(first.run_dir, resume=True); resumed.run()
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(resumed.run_dir / 'last.ckpt', resumed.identity)
        self.assertEqual(b['identity']['ce_reduction_mode'], BALANCED)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator'):
            self.fixture.fixture.assert_nested_equal(a[key], b[key])
        voxel = self.fixture.trainer(self.root / 'voxel'); voxel.run(stop_after=2)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            trainer(voxel.run_dir, resume=True)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            trainer(voxel.run_dir, resume=True, extend_to=6)
        for row in self.fixture.fixture.rows(resumed.run_dir):
            self.assertEqual(row['ce_reduction_mode'], BALANCED)
            if row['phase'] == 'train_monitor':
                for case in row['diagnostic_cases'].values():
                    self.assertIn('balanced_ce', case)

    def test_cli_balanced_synthetic_run_and_resume_identity(self):
        spec = importlib.util.spec_from_file_location('balanced_cli', ROOT / 'scripts/train.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        data = self.root / 'data'; data.mkdir(); fixtures.synthetic_pair(data, shape=(12, 12, 12))
        for folder in ('imagesTr', 'labelsTr'):
            shutil.copyfile(data / folder / 'amos_0001.nii.gz', data / folder / 'amos_0002.nii.gz')
        atomic_json(data / 'dataset.json', {'training': [dict(image=f'imagesTr/amos_{i:04}.nii.gz',
                    label=f'labelsTr/amos_{i:04}.nii.gz') for i in (1, 2)]})
        selection = json.loads((ROOT / 'configs/ct_stats.json').read_text())
        artifact = fixtures.create_development(fixtures.training_manifest(data, selection), 17, 1)
        split = self.root / 'split.json'; fixtures.write_split(split, artifact)
        cfg = json.loads((ROOT / 'configs/train_backbone_only_overfit_balanced_ce.json').read_text())
        for key in ('model_config', 'baseline_config', 'selection_config'):
            cfg[key] = str(ROOT / 'configs' / cfg[key])
        cfg['runtime'].update(backend='cpu', device='cpu', cpu_threads=1)
        cfg['data'].update(train_cases=None, train_limit=1)
        cfg['training'].update(max_steps=2, validation_every=1)
        path = self.root / 'config.json'; atomic_json(path, cfg)
        run = self.root / 'run'
        argv = ['--config', str(path), '--data-root', str(data), '--split', str(split),
                '--run-dir', str(run), '--cpu-synthetic', '--quiet-console', '--no-tensorboard']
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv + ['--stop-after', '1']), 0)
            self.assertEqual(cli.main(argv + ['--resume', str(run / 'last.ckpt')]), 0)
        identity = json.loads((run / 'run.json').read_text())['identity']
        self.assertEqual(identity['ce_reduction_mode'], BALANCED)
        self.assertEqual(load_checkpoint(run / 'last.ckpt', identity)['progress']['global_step'], 2)
