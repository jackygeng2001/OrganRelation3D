"""Hierarchical foreground CE only; no change to model or 15-class Dice."""
import json
import unittest

import torch
import test_balanced_ce as balanced
from organ_relation.losses import (JointLoss, SegmentationLoss, foreground_class_ce_means,
                                   foreground_background_ce_means)
from organ_relation.models.segmentor import SegmentorOutput
from organ_relation.metrics import hard_dice
from organ_relation.training.engine import final_diagnostic_metrics, joint_ce_diagnostic_metrics
from organ_relation.training.state import load_checkpoint, digest

ROOT = balanced.ROOT
OPTIONS = dict(epsilon=1e-6, ce_reduction_mode=balanced.BALANCED,
               foreground_ce_reduction='class_macro_mean')


class ForegroundMacroTests(unittest.TestCase):
    def setUp(self):
        self.fixture = balanced.BalancedCETests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def test_hand_calculation_unequal_voxels_equal_votes_absent_excluded(self):
        losses = torch.tensor([[1., 1., 2., 2., 2., 8.]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([[0, 0, 1, 1, 1, 15]])
        means, macro, count = foreground_class_ce_means(losses, labels)
        self.assertEqual(count.item(), 2)
        self.assertEqual(means[0, 0].item(), 2)
        self.assertEqual(means[0, 14].item(), 8)
        self.assertTrue(torch.isnan(means[0, 1:14]).all())
        self.assertEqual(macro.item(), 5)  # (2+8)/2, NOT (2+2+2+8)/4 or /15.
        macro.sum().backward()
        torch.testing.assert_close(losses.grad, torch.tensor([[0., 0., 1/6, 1/6, 1/6, .5]], dtype=torch.float64))

    def test_macro_formula_batch_reference_unchanged_all_15_dice(self):
        logits, label = self.fixture.example()
        actual = SegmentationLoss(**OPTIONS)(logits, label)
        old = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=balanced.BALANCED)(logits, label)
        terms = -torch.log(logits.softmax(1).gather(1, label.unsqueeze(1)).squeeze(1) + 1e-6)
        ce = []
        for b in range(2):
            present = label[b].unique().tolist(); present.remove(0)
            per_class = torch.stack([terms[b][label[b] == c].mean() for c in present])
            ce.append(.5 * terms[b][label[b] == 0].mean() + .5 * per_class.mean())
        ce = torch.stack(ce)
        torch.testing.assert_close(actual.final.ce, ce, rtol=1e-14, atol=1e-14)
        self.assertTrue(torch.equal(actual.final.dice_per_class, old.final.dice_per_class))
        self.assertEqual(actual.final.dice_per_class.shape, (2, 15))
        expected = (ce + old.final.dice_loss).mean()
        torch.testing.assert_close(actual.total, expected)
        ga = torch.autograd.grad(actual.total, logits, retain_graph=True)[0]
        gb = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(ga, gb, rtol=1e-12, atol=1e-12)
        # Each case has its own present-class set; batch pooling is different.
        pooled_macro = torch.stack([terms[label == c].mean() for c in label.unique().tolist() if c]).mean()
        pooled_ce = .5 * terms[label == 0].mean() + .5 * pooled_macro
        self.assertGreater(abs(pooled_ce.item() - ce.mean().item()), 1e-3)

    def test_default_voxel_foreground_historical_ce_and_gradients_bit_exact(self):
        for dtype in (torch.float32, torch.float64):
            f, y = self.fixture.example(dtype)
            args = dict(epsilon=1e-6, ce_reduction_mode=balanced.BALANCED)
            for bg_weight, fg_weight in ((.5, .5), (.7, .3)):
                opts = dict(args, ce_background_weight=bg_weight, ce_foreground_weight=fg_weight)
                a = SegmentationLoss(**opts)(f, y)
                b = SegmentationLoss(**opts, foreground_ce_reduction='voxel_mean')(f, y)
                matched = f.softmax(1).flatten(2).gather(1, y.flatten(1).unsqueeze(1)).squeeze(1)
                bg, fg = foreground_background_ce_means(-torch.log(matched + 1e-6), y.flatten(1) > 0)
                historical = bg_weight * bg + fg_weight * fg
                self.assertTrue(torch.equal(a.final.ce, historical))
                self.assertTrue(torch.equal(a.total, b.total))
                ga = torch.autograd.grad(a.total, f, retain_graph=True)[0]
                gb = torch.autograd.grad(b.total, f, retain_graph=True)[0]
                self.assertTrue(torch.equal(ga, gb))

    def test_joint_coarse_final_macro_and_upsample_before_softmax(self):
        f, y = self.fixture.example()
        c = torch.randn(2, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        actual = JointLoss(**OPTIONS, lambda_c=.5, align_corners=False)(c, f, y)
        up = torch.nn.functional.interpolate(c, size=y.shape[1:], mode='trilinear', align_corners=False)
        expected_c = SegmentationLoss(**OPTIONS)(up, y)
        expected_f = SegmentationLoss(**OPTIONS)(f, y)
        for a, b in ((actual.coarse, expected_c.final), (actual.final, expected_f.final)):
            for x, z in zip(a, b):
                self.assertTrue(torch.equal(x, z))
        expected = (expected_f.per_case + .5 * expected_c.per_case).mean()
        self.assertTrue(torch.equal(actual.total, expected))
        ga = torch.autograd.grad(actual.total, (c, f), retain_graph=True)
        gb = torch.autograd.grad(expected, (c, f), retain_graph=True)
        for x, z in zip(ga, gb):
            self.assertTrue(torch.equal(x, z))
        wrong_p = torch.nn.functional.interpolate(c.softmax(1), size=y.shape[1:], mode='trilinear', align_corners=False)
        wrong = SegmentationLoss(**OPTIONS)(wrong_p.log(), y)
        self.assertGreater((actual.coarse.ce - wrong.final.ce).abs().max().item(), 1e-3)

    def test_macro_fp64_gradcheck_and_empty_group_rejection(self):
        f = torch.randn(1, 16, 1, 1, 4, dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[[[0, 1, 1, 15]]]])
        criterion = SegmentationLoss(**OPTIONS)
        self.assertTrue(torch.autograd.gradcheck(lambda x: criterion(x, y).total, (f,)))
        for value in (0, 1):
            with self.assertRaisesRegex(ValueError, 'background and foreground'):
                criterion(f, torch.full_like(y, value))
        for bad in ('invalid', None):
            with self.assertRaisesRegex(ValueError, 'foreground_ce_reduction'):
                SegmentationLoss(epsilon=1e-6, foreground_ce_reduction=bad)
        with self.assertRaisesRegex(ValueError, 'requires'):
            SegmentationLoss(epsilon=1e-6, foreground_ce_reduction='class_macro_mean')

    def test_monitor_per_class_null_macro_and_actual_weighted_ce(self):
        f, y = self.fixture.example(); f, y = f[:1], y[:1]
        criterion = SegmentationLoss(**OPTIONS)
        record = final_diagnostic_metrics(f, y, criterion, hard_dice(f.argmax(1)[0], y[0]))
        self.assertEqual(record['foreground_ce_reduction'], 'class_macro_mean')
        self.assertEqual(record['present_foreground_class_count'], 1)
        self.assertEqual(len(record['per_class_ce_mean']), 15)
        self.assertEqual(record['per_class_ce_mean'][1:], [None] * 14)
        self.assertEqual(record['CE_fg_macro'], record['per_class_ce_mean'][0])
        self.assertEqual(record['weighted_ce'], criterion(f, y).final.ce.item())
        c = torch.randn(1, 16, 1, 1, 2, dtype=torch.float64)
        joint = JointLoss(**OPTIONS, lambda_c=.5, align_corners=False)
        records = joint_ce_diagnostic_metrics(SegmentorOutput(c, f), y, joint)
        result = joint(c, f, y)
        for name in ('coarse', 'final'):
            self.assertEqual(records[name]['weighted_ce'], getattr(result, name).ce.item())
            self.assertEqual(records[name]['present_foreground_class_count'], 1)

    def test_config_only_foreground_reduction_differs_and_cli_resume(self):
        old = json.loads((ROOT / 'configs/train_backbone_only_overfit_balanced_ce.json').read_text())
        new = json.loads((ROOT / 'configs/train_backbone_only_overfit_fg_class_macro.json').read_text())
        self.assertEqual(new.pop('foreground_ce_reduction'), 'class_macro_mean')
        self.assertEqual(new, old)
        self.fixture.cli_balanced_run('train_backbone_only_overfit_fg_class_macro.json')

    def test_resume_macro_exact_trajectory_and_reject_changed_reduction(self):
        def trainer(path, **kw):
            return self.fixture.fixture.trainer(path, ce_reduction_mode=balanced.BALANCED,
                foreground_ce_reduction='class_macro_mean', **kw)
        whole = trainer(self.root / 'whole'); whole.run()
        first = trainer(self.root / 'split'); first.run(stop_after=2)
        resumed = trainer(first.run_dir, resume=True); resumed.run()
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(resumed.run_dir / 'last.ckpt', resumed.identity)
        self.assertEqual(b['identity']['foreground_ce_reduction'], 'class_macro_mean')
        for key in ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator'):
            self.fixture.fixture.fixture.assert_nested_equal(a[key], b[key])
        wrong = dict(resumed.identity, foreground_ce_reduction='voxel_mean')
        for extension in (False, True):
            with self.assertRaises(ValueError):
                load_checkpoint(resumed.run_dir / 'last.ckpt', wrong, extension=extension)

    def test_legacy_missing_foreground_mode_preserves_origin_and_hash(self):
        trainer = self.fixture.fixture.trainer
        first = trainer(self.root / 'legacy', ce_reduction_mode=balanced.BALANCED)
        first.run(stop_after=2)
        before = (first.run_dir / 'run.json').read_bytes()
        origin_hash = digest(first.identity)
        resumed = trainer(first.run_dir, ce_reduction_mode=balanced.BALANCED,
                          foreground_ce_reduction='voxel_mean', resume=True)
        resumed.run()
        self.assertEqual((first.run_dir / 'run.json').read_bytes(), before)
        self.assertEqual(digest(resumed.origin_identity), origin_hash)


if __name__ == '__main__':
    unittest.main()
