"""Independent step saves/presentation with exact mid-epoch CPU recovery."""
import io
import unittest
from unittest.mock import MagicMock, patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from test_development import TinyReference
from test_training_v2 import config
from test_monitoring import Terminal, train_row
from organ_relation.training.console import TrainingConsole
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.monai_reference import MonaiReferenceLoss
from organ_relation.training.state import seed_all, load_checkpoint, capture_rng


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def trainer(self, name, *, resume=False, count=160, intervals=True):
        cfg = config(); options = cfg['training']
        if not intervals:
            options.pop('checkpoint_every_steps'); options.pop('console_every_steps')
        seed_all(options['seed'])
        model = TinyReference()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        class Cases(fixtures.RandomCases):
            def __len__(self): return count
            def __getitem__(self, index): return super().__getitem__(index % 3)
        class Dev(Cases):
            def __len__(self): return 40
        path = self.root / name
        identity = dict(mode=cfg['mode'], training=options, model='tiny', loss=cfg['loss'],
                        optimizer=cfg['optimizer'], preprocessing='synthetic', provenance='test',
                        data={'manifest_hash': 'synthetic'})
        return Trainer(model, MonaiReferenceLoss(cfg['loss']), optimizer, Cases(), [f't{i}' for i in range(count)],
            device='cpu', options=options, identity=identity, run_dir=path,
            validation_dataset=Dev(), validation_case_ids=[f'v{i}' for i in range(40)],
            resume=path/'last.ckpt' if resume else None, tensorboard=True,
            console=TrainingConsole(stream=io.StringIO()))

    def checkpoint(self, trainer):
        return load_checkpoint(trainer.run_dir/'last.ckpt', trainer.identity)

    def compare_states(self, a, b):
        for key in ('model', 'optimizer', 'rng', 'progress', 'sampler_generator', 'loader_generator',
                    'development_state', 'early_stopping'):
            self.f.assert_nested_equal(a[key], b[key])

    def test_config_fields_independent_positive_and_validation_unchanged(self):
        for gated in (False, True):
            opt = config(gated)['training']; validate_options(opt)
            self.assertEqual(opt['checkpoint_every_steps'], 5)
            self.assertEqual(opt['console_every_steps'], 5)
            self.assertEqual(opt['cadence_unit'], 'epoch')
            self.assertEqual(opt['checkpoint_every'], 1)
            self.assertEqual(opt['validation_every'], 5)
            for key in ('checkpoint_every_steps', 'console_every_steps'):
                for bad in (0, -1, True, 2.5, None):
                    with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                        validate_options(dict(opt, **{key: bad}))

    def test_crash_after_step7_resumes_step5_then_step10_exact_trajectory(self):
        whole = self.trainer('whole'); snapshots = {}; original = whole.checkpoint
        def saved():
            original(); snapshots[whole.state['global_step']] = self.checkpoint(whole)
        with patch.object(whole, 'checkpoint', side_effect=saved):
            self.f.quiet_run(whole, stop_after=12)
        self.assertEqual(list(snapshots), [5, 10, 12])

        broken = self.trainer('broken'); batch = broken._batch
        def fail(*args, **kwargs):
            if broken.state['global_step'] == 7:
                raise RuntimeError('synthetic interruption after seven committed train rows')
            return batch(*args, **kwargs)
        with patch.object(broken, '_batch', side_effect=fail), self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
            self.f.quiet_run(broken)
        ck5 = self.checkpoint(broken)
        self.assertEqual(ck5['progress']['global_step'], 5)
        self.assertEqual(ck5['progress']['epoch'], 0)
        self.assertEqual(ck5['progress']['cursor'], 5)
        self.assertEqual(len(self.f.rows(broken.run_dir)), 7)
        self.assertFalse((broken.run_dir/'best-dev.ckpt').exists())
        self.compare_states(snapshots[5], ck5)

        resumed = self.trainer('broken', resume=True)
        self.f.assert_nested_equal(ck5['rng'], capture_rng(resumed.device))
        self.assertEqual(len(self.f.rows(resumed.run_dir)), 5)  # Uncommitted tail removed.
        self.f.assert_nested_equal(ck5['progress'], resumed.state)
        self.f.quiet_run(resumed, stop_after=5)
        self.compare_states(snapshots[10], self.checkpoint(resumed))
        again = self.trainer('broken', resume=True)
        self.f.quiet_run(again, stop_after=2)
        self.compare_states(snapshots[12], self.checkpoint(again))
        a = [r for r in self.f.rows(whole.run_dir) if r['phase'] == 'train']
        b = [r for r in self.f.rows(again.run_dir) if r['phase'] == 'train']
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            for key in ('epoch', 'global_step', 'case_id', 'total_loss', 'final', 'branch_metrics'):
                self.assertEqual(x[key], y[key])
        self.assertFalse(any(r['phase'] == 'internal_dev' for r in self.f.rows(again.run_dir)))
        events = EventAccumulator(str(again.board.directory)).Reload()
        self.assertEqual(events.Tags()['scalars'], [])  # No step-level TensorBoard additions.

    def test_step_saves_and_console_do_not_change_training(self):
        legacy = self.trainer('epoch_only', intervals=False)
        self.f.quiet_run(legacy, stop_after=12)
        current = self.trainer('step_saves')
        self.f.quiet_run(current, stop_after=12)
        self.compare_states(self.checkpoint(legacy), self.checkpoint(current))

    def test_non_multiple_epoch_end_still_saves_without_creating_best(self):
        trainer = self.trainer('seven_cases', count=7); original = trainer.checkpoint; steps = []
        def saved():
            original(); steps.append(trainer.state['global_step'])
        with patch.object(trainer, 'checkpoint', side_effect=saved):
            self.f.quiet_run(trainer, stop_after=8)
        self.assertEqual(steps, [5, 7, 8])
        self.assertFalse((trainer.run_dir/'best-dev.ckpt').exists())
        self.assertEqual(trainer.development_state['validation_history'], [])

    def test_epoch3_cursor47_resumes_next_case_without_reshuffle(self):
        first = self.trainer('cursor47'); self.f.quiet_run(first, stop_after=367)
        ck = self.checkpoint(first)
        self.assertEqual((ck['progress']['epoch'], ck['progress']['cursor']), (2, 47))
        next_case = first.case_ids[ck['progress']['order'][47]]
        resumed = self.trainer('cursor47', resume=True)
        self.f.assert_nested_equal(ck['progress'], resumed.state)
        self.f.assert_nested_equal(ck['sampler_generator'], resumed.sampler_generator.get_state())
        self.f.quiet_run(resumed, stop_after=1)
        row = self.f.rows(resumed.run_dir)[-1]
        self.assertEqual((row['epoch'], row['global_step'], row['case_id']), (3, 368, next_case))
        self.assertEqual(resumed.state['order'], ck['progress']['order'])

    def test_console_every_five_steps_fields_gamma_and_baseline_no_fake_data(self):
        for tty in (False, True):
            for gated in (False, True):
                with self.subTest(tty=tty, gated=gated):
                    stream = Terminal(tty); console = TrainingConsole(stream=stream)
                    with patch.object(console, '_bar', return_value=MagicMock()):
                        console.start({}, config()['training'], list(range(160)), 80000, 500,
                                      dict(epoch=0, global_step=0, cursor=0))
                        console.begin_step(1, 0)
                        for step in range(1, 11):
                            row = train_row(step); row['epoch'] = 1
                            row['total_loss'] = float(step)
                            row['final']['segmentation'] = 3.1
                            if gated:
                                row['coarse']['segmentation'] = 2.05
                                row['relation_scale'] = dict(gamma=.0973, scaled_writeback_to_feature_norm=.142)
                            else:
                                row.pop('coarse')
                            console.train_step(row)
                        console.close()
                    text = stream.getvalue()
                    self.assertEqual(text.count('[Train summary]'), 2)
                    for value in ('5 / 80000', '10 / 80000', '5 / 160', '10 / 160',
                                  'Mean Loss (5)', '3.0000', '8.0000', 'Final Loss', 'LR', 'GPU Peak', 'Step Mean', 'Train ETA'):
                        self.assertIn(value, text)
                    self.assertEqual('Gamma' in text, gated)
                    self.assertEqual('Writeback/F' in text, gated)
                    self.assertEqual('Coarse Loss' in text, gated)
                    if gated:
                        self.assertIn('0.0973', text); self.assertIn('0.142', text)


if __name__ == '__main__': unittest.main()
