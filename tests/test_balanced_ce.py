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
from organ_relation.training.engine import final_diagnostic_metrics, joint_ce_diagnostic_metrics
from organ_relation.models.segmentor import SegmentorOutput
from organ_relation.metrics import hard_dice
from organ_relation.training.state import atomic_json, load_checkpoint, digest

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

    def test_weight_defaults_explicit_5050_exact_outputs_and_gradients(self):
        weights = dict(ce_background_weight=.5, ce_foreground_weight=.5)
        for dtype in (torch.float32, torch.float64):
            f, y = self.example(dtype)
            c = torch.randn(2, 16, 1, 1, 2, dtype=dtype, requires_grad=True)
            for kind in ('final', 'joint'):
                cls = SegmentationLoss if kind == 'final' else JointLoss
                args = dict(epsilon=1e-6, ce_reduction_mode=BALANCED)
                if kind == 'joint':
                    args.update(lambda_c=.5, align_corners=False)
                inputs = (f, y) if kind == 'final' else (c, f, y)
                a, b = cls(**args)(*inputs), cls(**args, **weights)(*inputs)
                self.assertTrue(torch.equal(a.total, b.total))
                for name in ('final',) if kind == 'final' else ('coarse', 'final'):
                    for x, z in zip(getattr(a, name), getattr(b, name)):
                        self.assertTrue(torch.equal(x, z))
                tensors = (f,) if kind == 'final' else (c, f)
                ga = torch.autograd.grad(a.total, tensors, retain_graph=True)
                gb = torch.autograd.grad(b.total, tensors, retain_graph=True)
                for x, z in zip(ga, gb):
                    self.assertTrue(torch.equal(x, z))

    def test_weighted_7030_formula_gradients_dice_and_joint_combination(self):
        f, y = self.example()
        kwargs = dict(epsilon=1e-6, ce_reduction_mode=BALANCED,
                      ce_background_weight=.7, ce_foreground_weight=.3)
        actual = SegmentationLoss(**kwargs)(f, y)
        terms = -torch.log(f.softmax(1).gather(1, y.unsqueeze(1)).squeeze(1) + 1e-6)
        ce = torch.stack([.7 * terms[b][y[b] == 0].mean() + .3 * terms[b][y[b] > 0].mean()
                          for b in range(2)])
        default = SegmentationLoss(epsilon=1e-6, ce_reduction_mode=BALANCED)(f, y)
        self.assertTrue(torch.equal(default.final.dice_loss, actual.final.dice_loss))
        reference = (ce + default.final.dice_loss).mean()
        torch.testing.assert_close(actual.total, reference, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(actual.total, f, retain_graph=True)[0],
                                   torch.autograd.grad(reference, f, retain_graph=True)[0])
        c = torch.randn(2, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        joint = JointLoss(**kwargs, lambda_c=.5, align_corners=False)(c, f, y)
        up = torch.nn.functional.interpolate(c, size=y.shape[1:], mode='trilinear', align_corners=False)
        coarse = SegmentationLoss(**kwargs)(up, y)
        torch.testing.assert_close(joint.total, (actual.per_case + .5 * coarse.per_case).mean())
        for branch, expected in ((joint.final, actual.final), (joint.coarse, coarse.final)):
            for x, z in zip(branch, expected):
                self.assertTrue(torch.equal(x, z))

    def test_invalid_group_weights_rejected(self):
        for bg, fg in ((0, 1), (1, 0), (-.1, 1.1), (.7, .5), (.3, .3),
                       (float('nan'), .5), (.5, float('inf')), (True, .5), ('0.5', .5), (None, .5)):
            for cls in (SegmentationLoss, JointLoss):
                args = dict(epsilon=1e-6, ce_reduction_mode=BALANCED,
                            ce_background_weight=bg, ce_foreground_weight=fg)
                if cls is JointLoss:
                    args.update(lambda_c=.5, align_corners=False)
                with self.assertRaisesRegex(ValueError, 'CE group weights'):
                    cls(**args)

    def test_7030_config_only_weights_differ(self):
        old = json.loads((ROOT / 'configs/train_backbone_only_overfit_balanced_ce.json').read_text())
        new = json.loads((ROOT / 'configs/train_backbone_only_overfit_bg70_fg30.json').read_text())
        self.assertEqual(new.pop('ce_background_weight'), .7)
        self.assertEqual(new.pop('ce_foreground_weight'), .3)
        self.assertEqual(new, old)

    def test_legacy_5050_identity_defaults_preserve_origin_hash_run_and_trajectory(self):
        weights = dict(ce_background_weight=.5, ce_foreground_weight=.5)
        first = self.fixture.trainer(self.root / 'legacy', ce_reduction_mode=BALANCED)
        first.run(stop_after=2)
        path = first.run_dir / 'last.ckpt'
        raw, run = path.read_bytes(), (first.run_dir / 'run.json').read_bytes()
        origin_hash = digest(first.identity)
        resumed = self.fixture.trainer(first.run_dir, resume=True, ce_reduction_mode=BALANCED, ce_weights=weights)
        self.assertEqual(path.read_bytes(), raw)  # Comparison never rewrites checkpoint.
        self.assertEqual(digest(resumed.origin_identity), origin_hash)
        resumed.run()
        self.assertEqual((first.run_dir / 'run.json').read_bytes(), run)
        whole = self.fixture.trainer(self.root / 'whole', ce_reduction_mode=BALANCED, ce_weights=weights)
        whole.run()
        a = load_checkpoint(path, resumed.identity)
        b = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        self.assertEqual(digest(a['origin_identity']), origin_hash)
        for key in ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator'):
            self.fixture.fixture.assert_nested_equal(a[key], b[key])
        for bad in (dict(resumed.identity, ce_background_weight=.7, ce_foreground_weight=.3),
                    dict(resumed.identity, provenance='different source'),
                    dict(resumed.identity, optimizer='different optimizer')):
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_checkpoint(path, bad)
            with self.assertRaises(ValueError):
                load_checkpoint(path, bad, extension=True)

    def test_weighted_7030_resume_exact_and_monitor_weighted_ce(self):
        weights = dict(ce_background_weight=.7, ce_foreground_weight=.3)
        def trainer(path, **kwargs):
            return self.fixture.trainer(path, ce_reduction_mode=BALANCED, ce_weights=weights, **kwargs)
        whole = trainer(self.root / 'whole'); whole.run()
        first = trainer(self.root / 'split'); first.run(stop_after=2)
        resumed = trainer(first.run_dir, resume=True); resumed.run()
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(resumed.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator'):
            self.fixture.fixture.assert_nested_equal(a[key], b[key])
        for row in self.fixture.fixture.rows(resumed.run_dir):
            for name, value in weights.items():
                self.assertEqual(row[name], value)
            if row['phase'] == 'train_monitor':
                for case in row['diagnostic_cases'].values():
                    self.assertAlmostEqual(case['weighted_ce'], .7*case['CE_bg_mean']+.3*case['CE_fg_mean'], places=6)
                    self.assertAlmostEqual(case['balanced_ce'], .5*case['CE_bg_mean']+.5*case['CE_fg_mean'], places=6)

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
        self.cli_balanced_run('train_backbone_only_overfit_balanced_ce.json')

    def test_cli_joint_balanced_full_segmentor_monitor_checkpoint_and_resume(self):
        self.cli_balanced_run('train_single_case_overfit_balanced_ce.json')

    def test_cli_weighted_7030_identity_and_resume(self):
        self.cli_balanced_run('train_backbone_only_overfit_bg70_fg30.json')

    def cli_balanced_run(self, config_name):
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
        cfg = json.loads((ROOT / 'configs' / config_name).read_text())
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
        self.assertEqual(identity['foreground_ce_reduction'], cfg.get('foreground_ce_reduction', 'voxel_mean'))
        for name in ('ce_background_weight', 'ce_foreground_weight'):
            self.assertEqual(identity[name], cfg.get(name, .5))
        for row in self.fixture.fixture.rows(run):
            for name in ('ce_background_weight', 'ce_foreground_weight'):
                self.assertEqual(row[name], cfg.get(name, .5))
        self.assertEqual(load_checkpoint(run / 'last.ckpt', identity)['progress']['global_step'], 2)
        if cfg.get('mode', 'organ_relation_joint') == 'organ_relation_joint':
            self.assertEqual(identity['loss']['lambda_c'], .5)
            rows = self.fixture.fixture.rows(run)
            for row in rows:
                self.assertEqual(row['mode'], 'organ_relation_joint')
                self.assertEqual(row['ce_reduction_mode'], BALANCED)
                if row['phase'] == 'train_monitor':
                    for case in row['ce_diagnostic_cases'].values():
                        self.assertEqual(set(case), {'coarse', 'final'})
                        for branch in case.values():
                            self.assertEqual(set(branch), {'CE_bg_mean', 'CE_fg_mean', 'balanced_ce',
                                                           'weighted_ce', 'ce_background_weight', 'ce_foreground_weight',
                                                           'foreground_ce_reduction', 'present_foreground_class_count',
                                                           'per_class_ce_mean', 'CE_fg_macro'})
            cfg['ce_reduction_mode'] = 'voxel_mean'; atomic_json(path, cfg)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(argv + ['--resume', str(run / 'last.ckpt')]), 2)

    def test_joint_config_only_reduction_and_200_step_difference(self):
        old = json.loads((ROOT / 'configs/train_single_case_overfit.json').read_text())
        new = json.loads((ROOT / 'configs/train_single_case_overfit_balanced_ce.json').read_text())
        self.assertEqual(new.pop('ce_reduction_mode'), BALANCED)
        self.assertEqual(new['training']['max_steps'], 200)
        new['training']['max_steps'] = old['training']['max_steps']
        self.assertEqual(new, old)

    def test_joint_default_historical_vector_formula_and_gradients_bit_exact(self):
        # Frozen pre-mode vector operations, independent of the current reducer.
        def old_branch(logits, label):
            p = logits.softmax(1)
            flat, indices = p.flatten(2), label.flatten(1)
            matched = flat.gather(1, indices.unsqueeze(1)).squeeze(1)
            ce = -torch.log(matched + 1e-6).mean(1)
            count = label.new_zeros(label.shape[0], 16).scatter_add(
                1, indices, torch.ones_like(indices)).to(logits.dtype)
            inter = p.new_zeros(p.shape[0], 16).scatter_add(1, indices, matched)
            dice = (2 * inter[:, 1:] + 1e-6) / (flat.sum(2)[:, 1:] + count[:, 1:] + 1e-6)
            dl = 1 - dice.mean(1)
            return ce, dice, dl, ce + dl
        for dtype in (torch.float32, torch.float64):
            final, label = self.example(dtype)
            coarse = torch.randn(2, 16, 1, 1, 2, dtype=dtype, requires_grad=True)
            for align in (False, True):
                args = dict(epsilon=1e-6, lambda_c=.5, align_corners=align)
                actual = JointLoss(**args)(coarse, final, label)
                explicit = JointLoss(**args, ce_reduction_mode='voxel_mean')(coarse, final, label)
                up = torch.nn.functional.interpolate(coarse, size=label.shape[1:], mode='trilinear', align_corners=align)
                rc, rf = old_branch(up, label), old_branch(final, label)
                for branch, reference in ((actual.coarse, rc), (actual.final, rf)):
                    for a, b in zip(branch, reference):
                        self.assertTrue(torch.equal(a, b))
                ref = (rf[3] + .5 * rc[3]).mean()
                self.assertTrue(torch.equal(actual.total, ref))
                self.assertTrue(torch.equal(actual.total, explicit.total))
                ga = torch.autograd.grad(actual.total, (coarse, final), retain_graph=True)
                gb = torch.autograd.grad(ref, (coarse, final), retain_graph=True)
                for a, b in zip(ga, gb):
                    self.assertTrue(torch.equal(a, b))

    def test_joint_balanced_both_branches_reference_lambda_batch_and_resize_order(self):
        final, label = self.example()
        coarse = torch.randn(2, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        for weight in (.5, .37):
            criterion = JointLoss(epsilon=1e-6, lambda_c=weight, align_corners=False, ce_reduction_mode=BALANCED)
            actual = criterion(coarse, final, label)
            up = torch.nn.functional.interpolate(coarse, size=label.shape[1:], mode='trilinear', align_corners=False)
            references = []
            original = JointLoss(epsilon=1e-6, lambda_c=weight, align_corners=False)(coarse, final, label)
            for logits, branch, old in ((up, actual.coarse, original.coarse),
                                        (final, actual.final, original.final)):
                p = logits.softmax(1)
                per_case = []
                for b in range(2):
                    loss = -torch.log(p[b].gather(0, label[b].unsqueeze(0)).squeeze(0) + 1e-6)
                    per_case.append(.5 * loss[label[b] == 0].mean() + .5 * loss[label[b] > 0].mean())
                ce = torch.stack(per_case)
                torch.testing.assert_close(branch.ce, ce, rtol=1e-14, atol=1e-14)
                self.assertTrue(torch.equal(branch.dice_per_class, old.dice_per_class))
                self.assertTrue(torch.equal(branch.dice_loss, old.dice_loss))
                references.append(ce + old.dice_loss)
            reference = references[1] + weight * references[0]
            torch.testing.assert_close(actual.per_case, reference)
            torch.testing.assert_close(actual.total, reference.mean())
            ga = torch.autograd.grad(actual.total, (coarse, final), retain_graph=True)
            gb = torch.autograd.grad(reference.mean(), (coarse, final), retain_graph=True)
            for a, b in zip(ga, gb):
                torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
            wrong = torch.nn.functional.interpolate(coarse.softmax(1), size=label.shape[1:],
                                                    mode='trilinear', align_corners=False)
            wrong_loss = -torch.log(wrong.gather(1, label.unsqueeze(1)).squeeze(1) + 1e-6)
            wrong_ce = torch.stack([.5*wrong_loss[b][label[b] == 0].mean()
                                   + .5*wrong_loss[b][label[b] > 0].mean() for b in range(2)])
            self.assertGreater((wrong_ce - actual.coarse.ce).abs().max().item(), 1e-3)

    def test_joint_monitor_both_branch_group_values_match_actual_ce(self):
        final, label = self.example()
        final, label = final[:1], label[:1]
        coarse = torch.randn(1, 16, 1, 1, 2, dtype=torch.float64)
        criterion = JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False, ce_reduction_mode=BALANCED)
        record = joint_ce_diagnostic_metrics(SegmentorOutput(coarse, final), label, criterion)
        actual = criterion(coarse, final, label)
        for name in ('coarse', 'final'):
            self.assertEqual(record[name]['balanced_ce'], getattr(actual, name).ce.item())
            self.assertAlmostEqual(record[name]['balanced_ce'],
                                   .5 * record[name]['CE_bg_mean'] + .5 * record[name]['CE_fg_mean'])
        for bad in ('class_balanced', None):
            with self.assertRaisesRegex(ValueError, 'ce_reduction_mode'):
                JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False, ce_reduction_mode=bad)
