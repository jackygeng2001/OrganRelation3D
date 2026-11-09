"""按步保存与控制台节奏独立于 epoch 协议，epoch 中途断点可精确恢复（CPU）。"""
import io
import unittest
from unittest.mock import MagicMock, patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import support as fixtures
from test_monitoring import Terminal, train_row
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.console import TrainingConsole
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.state import seed_all, load_checkpoint, capture_rng

MAIN_CONFIG = 'configs/train_brats_year_fold_bbox_amp_bf16.json'
EARLY = dict(min_epochs=20, patience_epochs=20, min_delta=0.001, monitor='dev_mean_region_hard_dice')


def options(intervals=True):
    """与正式五折配置相同的节奏字段：每 epoch 保存、每 5 epoch 验证，另每 5 步保存/打印。"""
    opt = fixtures.training_options(cadence_unit='epoch', max_steps=None, max_epochs=500, checkpoint_every=1,
                                    validation_every=5, validation_role='internal_dev', early_stopping=dict(EARLY),
                                    checkpoint_every_steps=5, console_every_steps=5)
    if not intervals:
        opt.pop('checkpoint_every_steps'); opt.pop('console_every_steps')
    return opt


class CadenceTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def trainer(self, name, *, resume=False, count=20, intervals=True):
        cfg = fixtures.config(); opt = options(intervals)
        seed_all(opt['seed'])
        model = BrainTumorSegmentor(BrainTumorConfig(**cfg['model']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        class Cases(fixtures.RandomCases):
            def __len__(self): return count
            def __getitem__(self, index): return super().__getitem__(index % 3)
        class Dev(Cases):
            def __len__(self): return 40
        path = self.root / name
        identity = dict(mode='brats_regions', model=cfg['model'], loss=cfg['loss'], threshold=.5, training=opt,
                        optimizer='AdamW', preprocessing=cfg['preprocessing'], provenance='test',
                        data={'manifest_hash': 'synthetic'})
        return Trainer(model, RegionJointLoss(**cfg['loss']), optimizer, Cases(), [f't{i}' for i in range(count)],
            device='cpu', options=opt, identity=identity, run_dir=path,
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
        for opt in (options(), fixtures.json.loads((fixtures.ROOT / MAIN_CONFIG).read_text())['training']):
            validate_options(opt)
            self.assertEqual(opt['cadence_unit'], 'epoch')
            self.assertEqual(opt['checkpoint_every'], 1)
            self.assertEqual(opt['checkpoint_every_steps'], opt['console_every_steps'])
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
        # BraTS 每步写训练标量；断点续训后步号唯一连续，且尚未到验证节奏，没有验证标量。
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], list(range(1, 13)))
        self.assertFalse(any(tag.startswith('Val/') for tag in events.Tags()['scalars']))

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

    def test_epoch3_cursor7_resumes_next_case_without_reshuffle(self):
        first = self.trainer('cursor7'); self.f.quiet_run(first, stop_after=47)
        ck = self.checkpoint(first)
        self.assertEqual((ck['progress']['epoch'], ck['progress']['cursor']), (2, 7))
        next_case = first.case_ids[ck['progress']['order'][7]]
        resumed = self.trainer('cursor7', resume=True)
        self.f.assert_nested_equal(ck['progress'], resumed.state)
        self.f.assert_nested_equal(ck['sampler_generator'], resumed.sampler_generator.get_state())
        self.f.quiet_run(resumed, stop_after=1)
        row = self.f.rows(resumed.run_dir)[-1]
        self.assertEqual((row['epoch'], row['global_step'], row['case_id']), (3, 48, next_case))
        self.assertEqual(resumed.state['order'], ck['progress']['order'])

    def test_console_every_five_steps_fields_and_no_fake_data(self):
        for tty in (False, True):
            with self.subTest(tty=tty):
                stream = Terminal(tty); console = TrainingConsole(stream=stream)
                with patch.object(console, '_bar', return_value=MagicMock()):
                    console.start({}, options(), list(range(160)), 80000, 500, dict(epoch=0, global_step=0, cursor=0))
                    console.begin_step(1, 0)
                    for step in range(1, 11):
                        row = train_row(step); row['epoch'] = 1
                        row['total_loss'] = float(step)
                        row['final']['segmentation'] = 3.1
                        row['coarse']['segmentation'] = 2.05
                        console.train_step(row)
                    console.close()
                text = stream.getvalue()
                self.assertEqual(text.count('[Train summary]'), 2)
                for value in ('5 / 80000', '10 / 80000', '5 / 160', '10 / 160', 'Mean Loss (5)', '3.0000', '8.0000',
                              'Final Loss', 'Coarse Loss', '2.0500', 'LR', 'GPU Peak', 'Step Mean', 'Train ETA'):
                    self.assertIn(value, text)
                self.assertNotIn('Gamma', text)


if __name__ == '__main__': unittest.main()
