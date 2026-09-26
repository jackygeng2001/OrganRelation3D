"""CPU diagnostic acceptance; no real CT, GPU or alternative training loop."""
import contextlib
import importlib.util
import io
import json
import shutil
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from organ_relation.losses import JointLoss, SegmentationLoss
from organ_relation.metrics import hard_dice
from organ_relation.models.backbone import Encoder3D, Decoder3D
from organ_relation.models.backbone_only import BackboneOnly
from organ_relation.models.segmentor import Segmentor
from organ_relation.models.segmentor_config import SegmentorConfig
from organ_relation.training.engine import Trainer, final_diagnostic_metrics
from organ_relation.training.state import atomic_json, load_checkpoint, seed_all
from organ_relation.evaluation.progress import CaseLedger

ROOT = fixtures.ROOT


def model_config(name='segmentor_micro.json'):
    return SegmentorConfig(**json.loads((ROOT / 'configs' / name).read_text())['model'])


class BackboneOnlyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def test_matched_initialization_every_tensor_and_rng(self):
        for name in ('segmentor_micro.json', 'segmentor_3stage_probe_p1.json'):
            cfg = model_config(name)
            torch.manual_seed(20260925)
            full = Segmentor(cfg)
            after_full = torch.get_rng_state()
            torch.manual_seed(20260925)
            diagnostic = BackboneOnly(cfg)
            self.assertTrue(torch.equal(after_full, torch.get_rng_state()))
            for part in ('encoder', 'decoder'):
                self.fixture.assert_nested_equal(getattr(full, part).state_dict(),
                                                  getattr(diagnostic, part).state_dict())

    def test_only_existing_encoder_decoder_registered_and_odd_even_shapes(self):
        model = BackboneOnly(model_config())
        self.assertIsInstance(model.encoder, Encoder3D)
        self.assertIsInstance(model.decoder, Decoder3D)
        self.assertEqual(set(dict(model.named_children())), {'encoder', 'decoder'})
        self.assertTrue(all(n.startswith(('encoder.', 'decoder.')) for n in model.state_dict()))
        for shape in ((1, 1, 8, 10, 12), (2, 1, 9, 11, 13), (1, 1, 1, 3, 5)):
            image = torch.randn(shape)
            out = model(image)
            self.assertEqual(out._fields, ('final_logits',))
            self.assertEqual(out.final_logits.shape, (shape[0], 16, *shape[2:]))
            encoded = model.encoder(image)
            self.assertTrue(torch.equal(out.final_logits, model.decoder(encoded.deepest, encoded.skips)))
        with self.assertRaises(TypeError):
            model(image, label=torch.zeros(shape))

    def test_final_loss_bit_exact_to_joint_final_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(81)
            logits = torch.randn(2, 16, 3, 4, 5, dtype=dtype, requires_grad=True)
            label = torch.randint(16, (2, 3, 4, 5))
            coarse = torch.randn(2, 16, 2, 2, 2, dtype=dtype)
            joint = JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False)(coarse, logits, label)
            final = SegmentationLoss(epsilon=1e-6)(logits, label)
            for a, b in zip(joint.final, final.final):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(final.total, joint.final.segmentation.mean()))
            a = torch.autograd.grad(final.total, logits, retain_graph=True)[0]
            b = torch.autograd.grad(joint.final.segmentation.mean(), logits)[0]
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(torch.equal(final.total, joint.total))

    def test_fp64_final_loss_gradcheck(self):
        logits = torch.randn(1, 16, 1, 1, 2, dtype=torch.float64, requires_grad=True)
        label = torch.tensor([[[[0, 12]]]])
        loss = SegmentationLoss(epsilon=1e-6)
        self.assertTrue(torch.autograd.gradcheck(lambda x: loss(x, label).total, (logits,)))

    def test_backward_every_parameter_has_finite_nonzero_gradient(self):
        torch.manual_seed(20260925)
        model = BackboneOnly(model_config())
        image = torch.randn(2, 1, 9, 11, 13, requires_grad=True)
        label = torch.randint(16, (2, 9, 11, 13))
        SegmentationLoss(epsilon=1e-6)(model(image).final_logits, label).total.backward()
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(p.grad.abs().sum().item(), 0, name)
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertFalse(label.requires_grad)

    def test_invalid_model_and_loss_inputs_rejected(self):
        model = BackboneOnly(model_config())
        for image in (torch.zeros(1, 2, 3, 3, 3), torch.full((1, 1, 3, 3, 3), float('nan')),
                      torch.zeros(1, 1, 3, 3, 3, dtype=torch.int64)):
            with self.assertRaises(ValueError):
                model(image)
        loss = SegmentationLoss(epsilon=1e-6)
        logits = torch.randn(1, 16, 2, 2, 2)
        for label in (torch.zeros(1, 2, 2, 2), torch.full((1, 2, 2, 2), 16), torch.zeros(1, 1, 2, 2).long()):
            with self.assertRaises(ValueError):
                loss(logits, label)
        for epsilon in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                SegmentationLoss(epsilon=epsilon)

    def trainer(self, directory, *, resume=False, extend_to=None, steps=4, board=False):
        seed_all(712)
        model = BackboneOnly(model_config()).to(memory_format=torch.channels_last_3d)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        opts = fixtures.config()['training']
        opts.update(max_steps=steps, validation_every=2, diagnostics_every=1, checkpoint_every=2)
        identity = dict(mode='backbone_only_final', model=model_config().to_dict(),
                        training=opts, loss={'epsilon': 1e-6}, optimizer='AdamW',
                        preprocessing='synthetic', provenance='test', data={'manifest_hash': 'test'})
        identity = json.loads(json.dumps(identity))  # Same canonical identity as the CLI.
        data = fixtures.RandomCases()
        return Trainer(model, SegmentationLoss(epsilon=1e-6), optimizer, data, ['one', 'two', 'three'],
                       device='cpu', options=opts, identity=identity, run_dir=directory,
                       validation_dataset=data, validation_case_ids=['one', 'two', 'three'],
                       resume=directory / 'last.ckpt' if resume else None, tensorboard=board,
                       extend_to=extend_to)

    def test_resume_exact_trajectory_optimizer_rng_monitor_and_no_fake_coarse_events(self):
        whole = self.trainer(self.root / 'whole')
        self.fixture.quiet_run(whole)
        first = self.trainer(self.root / 'split', board=True)
        self.fixture.quiet_run(first, stop_after=2)
        resumed = self.trainer(first.run_dir, resume=True, board=True)
        self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(first.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator'):
            self.fixture.assert_nested_equal(a[key], b[key])
        for x, y in zip(self.fixture.rows(whole.run_dir), self.fixture.rows(first.run_dir)):
            for key in ('mode', 'phase', 'global_step', 'case_id', 'total_loss', 'final', 'diagnostics',
                        'metrics', 'diagnostic_cases'):
                self.assertEqual(x.get(key), y.get(key), key)
            self.assertEqual(y['mode'], 'backbone_only_final')
            self.assertNotIn('coarse', y)
            if y['phase'] == 'train':
                self.assertEqual(set(y['diagnostics']), {'encoder', 'decoder'})
            else:
                self.assertEqual(set(y['diagnostic_cases']), {'one', 'two', 'three'})
                self.assertEqual(len(y['metrics']['organs']), 15)
        events = EventAccumulator(str(resumed.board.directory), size_guidance={'scalars': 0}).Reload()
        self.assertFalse(resumed.board.failed)
        self.assertFalse(any('Coarse' in tag for tag in events.Tags()['scalars']))
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], [1, 2, 3, 4])
        self.assertEqual([e.step for e in events.Scalars('Monitor/FinalSoftDice_Mean')], [2, 4])
        self.assertIsNone(resumed.board.writer)

    def test_controlled_extension_matches_uninterrupted_diagnostic(self):
        whole = self.trainer(self.root / 'whole', steps=6)
        self.fixture.quiet_run(whole)
        first = self.trainer(self.root / 'extended')
        self.fixture.quiet_run(first)
        resumed = self.trainer(first.run_dir, resume=True, extend_to=6)
        self.fixture.quiet_run(resumed)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator'):
            a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)[key]
            b = load_checkpoint(resumed.run_dir / 'last.ckpt', resumed.identity)[key]
            self.fixture.assert_nested_equal(a, b)

    def test_pending_diagnostic_ledger_resumes_without_recomputing_completed_case(self):
        trainer = self.trainer(self.root / 'run')
        original = CaseLedger.commit
        def interrupt(ledger, case, data, record):
            if case == 'two':
                raise OSError('simulated interruption')
            return original(ledger, case, data, record)
        with patch.object(CaseLedger, 'commit', interrupt), self.assertRaises(OSError):
            self.fixture.quiet_run(trainer)
        resumed = self.trainer(trainer.run_dir, resume=True)
        written = []
        def track(ledger, case, data, record):
            written.append(case)
            return original(ledger, case, data, record)
        with patch.object(CaseLedger, 'commit', track):
            self.fixture.quiet_run(resumed, stop_after=1)
        self.assertEqual(written, ['two', 'three'])
        row = next(r for r in self.fixture.rows(trainer.run_dir) if r['phase'] == 'train_monitor')
        self.assertIn('final_soft_dice', row['diagnostic_cases']['one'])

    def test_diagnostic_counts_and_probabilities_hand_computed(self):
        probs = torch.full((1, 16, 1, 1, 4), .1 / 15, dtype=torch.float64)
        # Background correct; foreground correct; wrong foreground class; missed foreground.
        label = torch.tensor([[[[0, 1, 2, 3]]]])
        predicted = torch.tensor([[[[0, 1, 4, 0]]]])
        probs.scatter_(1, predicted.unsqueeze(1), .9)
        logits = probs.log()
        loss = SegmentationLoss(epsilon=1e-6)
        hard = hard_dice(predicted[0], label[0])
        result = final_diagnostic_metrics(logits, label, loss, hard)
        self.assertEqual(result['predicted_foreground_voxels'], 2)
        self.assertEqual(result['foreground_true_positive_voxels'], 1)
        self.assertEqual(result['gt_foreground_voxels'], 3)
        expected = (.9 + 2 * .1 / 15) / 3
        self.assertAlmostEqual(result['gt_foreground_true_class_mean_probability'], expected)
        self.assertAlmostEqual(result['gt_foreground_background_mean_probability'], expected)
        self.assertEqual(result['final_soft_dice'], loss(logits, label).final.dice_per_class.mean().item())
        empty = torch.zeros_like(label)
        result = final_diagnostic_metrics(logits, empty, loss, hard_dice(predicted[0], empty[0]))
        self.assertIsNone(result['gt_foreground_true_class_mean_probability'])
        self.assertIsNone(result['gt_foreground_background_mean_probability'])

    def test_200_step_config_semantics(self):
        cfg = json.loads((ROOT / 'configs/train_backbone_only_overfit.json').read_text())
        previous = json.loads((ROOT / 'configs/train_single_case_overfit.json').read_text())
        self.assertEqual(cfg['mode'], 'backbone_only_final')
        for key in ('model_config', 'baseline_config', 'selection_config', 'runtime', 'optimizer'):
            self.assertEqual(cfg[key], previous[key])
        self.assertEqual(cfg['data']['train_cases'], ['amos_0109'])
        self.assertEqual(cfg['data']['candidate'], 'B')
        self.assertIsNone(cfg['data']['train_limit'])
        self.assertEqual(cfg['training'], dict(previous['training'], max_steps=200))
        self.assertEqual([cfg['training'][key] for key in ('checkpoint_every', 'diagnostics_every', 'validation_every')], [25]*3)
        self.assertEqual(cfg['training']['seed'], 20260925)
        self.assertEqual(tuple(model_config(cfg['model_config']).backbone.channels), (8, 16, 32))

    def test_cli_synthetic_nifti_diagnostic_resume_and_mode_identity(self):
        spec = importlib.util.spec_from_file_location('diagnostic_cli', ROOT / 'scripts/train.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        data = self.root / 'data'; data.mkdir(); fixtures.synthetic_pair(data, shape=(12, 12, 12))
        for folder in ('imagesTr', 'labelsTr'):
            shutil.copyfile(data / folder / 'amos_0001.nii.gz', data / folder / 'amos_0002.nii.gz')
        atomic_json(data / 'dataset.json', {'training': [dict(image=f'imagesTr/amos_{i:04}.nii.gz',
                    label=f'labelsTr/amos_{i:04}.nii.gz') for i in (1, 2)]})
        selection = json.loads((ROOT / 'configs/ct_stats.json').read_text())
        artifact = fixtures.create_development(fixtures.training_manifest(data, selection), 17, 1)
        split = self.root / 'split.json'; fixtures.write_split(split, artifact)
        cfg = json.loads((ROOT / 'configs/train_backbone_only_overfit.json').read_text())
        for key in ('model_config', 'baseline_config', 'selection_config'):
            cfg[key] = str(ROOT / 'configs' / cfg[key])
        cfg['runtime'].update(backend='cpu', device='cpu', cpu_threads=1)
        cfg['data'].update(train_cases=None, train_limit=1)
        cfg['training'].update(max_steps=2, validation_every=1, diagnostics_every=1)
        path = self.root / 'config.json'; atomic_json(path, cfg)
        run = self.root / 'run'
        argv = ['--config', str(path), '--data-root', str(data), '--split', str(split),
                '--run-dir', str(run), '--cpu-synthetic', '--quiet-console']
        original = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv + ['--stop-after', '1']), 0)
            self.assertEqual(cli.main(argv + ['--resume', str(run / 'last.ckpt')]), 0)
        identity = json.loads((run / 'run.json').read_text())['identity']
        self.assertEqual(identity['mode'], 'backbone_only_final')
        self.assertEqual(identity['loss'], {'epsilon': 1e-6})
        self.assertEqual(identity['initialization'], 'retain_encoder_decoder_from_seeded_full_segmentor')
        self.assertEqual([r['global_step'] for r in self.fixture.rows(run) if r['phase'] == 'train'], [1, 2])
        for p, raw in original.items():
            self.assertEqual(p.read_bytes(), raw)
        cfg['mode'] = 'organ_relation_joint'; atomic_json(path, cfg)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(argv + ['--resume', str(run / 'last.ckpt')]), 2)


if __name__ == '__main__':
    unittest.main()
