"""按 epoch 的开发协议：验证节奏、best-dev 选择、待完成验证的断点复用与受控延长（CPU 合成数据）。"""
import copy
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import support as fixtures
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.brain_tumor import summarize_regions
from organ_relation.training.development import cadence_due
from organ_relation.training.engine import Trainer
from organ_relation.training.extension import resolve_horizon
from organ_relation.training.state import load_checkpoint, seed_all


class EpochProtocolTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def trainer(self, path, resume=False, extend_to=None, epochs=20):
        """三例训练、三例 internal dev；每 10 个 epoch 验证一次。"""
        seed_all(20260925)
        cfg = fixtures.config()
        options = fixtures.training_options(cadence_unit='epoch', max_steps=None, max_epochs=epochs,
                                            checkpoint_every=1, validation_every=10, validation_role='internal_dev',
                                            memory_format='contiguous')
        model = BrainTumorSegmentor(BrainTumorConfig(**cfg['model']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        ident = dict(mode='brats_regions', model=cfg['model'], loss=cfg['loss'], threshold=.5, training=options,
                     optimizer='AdamW', preprocessing=cfg['preprocessing'], provenance='test',
                     data={'manifest_hash': 'synthetic', 'split_hash': 'synthetic'})
        cases = fixtures.RandomCases()
        return Trainer(model, RegionJointLoss(**cfg['loss']), optimizer, cases, ['one', 'two', 'three'],
            device='cpu', options=options, identity=ident, run_dir=path,
            validation_dataset=cases, validation_case_ids=['one', 'two', 'three'],
            resume=path / 'last.ckpt' if resume else None, extend_to=extend_to, tensorboard=True)

    def test_cadence_exact_epochs_and_no_step_based_validation(self):
        opt = dict(cadence_unit='epoch', validation_every=5, checkpoint_every=1)
        vals, checkpoints = [], []
        for step in range(1, 80001):
            epoch = (step - 1) // 160 + 1
            end = step % 160 == 0
            if cadence_due(opt, 'validation_every', step, epoch, end): vals.append(epoch)
            if cadence_due(opt, 'checkpoint_every', step, epoch, end): checkpoints.append(step)
        self.assertEqual(vals, list(range(5, 501, 5)))
        self.assertEqual(checkpoints, list(range(160, 80001, 160)))

    def test_epochs_resume_best_history_jsonl_and_board(self):
        whole = self.trainer(self.root / 'whole')
        self.f.quiet_run(whole)
        first = self.trainer(self.root / 'resumed')
        self.f.quiet_run(first, stop_after=29)  # 第 10 个 epoch 验证之前的半个 epoch 处中断。
        second = self.trainer(first.run_dir, resume=True)
        self.f.quiet_run(second)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(second.run_dir / 'last.ckpt', second.identity)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator', 'development_state'):
            self.f.assert_nested_equal(a[key], b[key])
        rows = self.f.rows(second.run_dir)
        epochs = [r for r in rows if r['phase'] == 'train_epoch']
        vals = [r for r in rows if r['phase'] == 'internal_dev']
        self.assertEqual([r['epoch'] for r in epochs], list(range(1, 21)))
        self.assertEqual([r['epoch'] for r in vals], [10, 20])
        self.assertTrue(all(r['metrics']['cases'] == 3 for r in vals))
        best = load_checkpoint(second.run_dir / 'best-dev.ckpt', second.identity)
        self.assertEqual(best['development_state']['best_dev']['score'], max(r['metrics']['mean_case_dice'] for r in vals))
        self.assertEqual(best['development_state']['best_dev']['global_step'], best['progress']['global_step'])
        self.assertEqual(len(b['development_state']['validation_history']), 2)
        train = [r for r in rows if r['phase'] == 'train']
        self.assertTrue(any(r['diagnostics'] for r in train))  # epoch 节奏下只在每个 epoch 末尾记录梯度诊断
        self.assertIn('memory', train[0])
        self.assertEqual(set(vals[0]['region_diagnostic_cases']), {'one', 'two', 'three'})
        self.assertAlmostEqual(epochs[0]['epoch_metrics']['total_loss'], sum(r['total_loss'] for r in train[:3]) / 3)
        events = EventAccumulator(str(second.board.directory), size_guidance={'scalars': 0}).Reload()
        self.assertEqual([e.step for e in events.Scalars('TrainEpoch/Total_Loss')], list(range(3, 61, 3)))
        self.assertEqual([e.step for e in events.Scalars('Val/HardDice_Mean')], [30, 60])
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')], list(range(1, 61)))

    def test_pending_validation_ledger_resume_without_retraining(self):
        run = self.trainer(self.root / 'pending', epochs=10)
        original = run._batch
        def interrupt(dataset, index, *, validation=False):
            if validation and index == 1:
                raise RuntimeError('synthetic interruption')
            return original(dataset, index, validation=validation)
        with patch.object(run, '_batch', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                self.f.quiet_run(run)
        ck = load_checkpoint(run.run_dir / 'last.ckpt', run.identity)
        self.assertEqual(ck['progress']['global_step'], 30)
        self.assertTrue(ck['progress']['pending_validation'])
        resumed = self.trainer(run.run_dir, resume=True, epochs=10)
        with patch.object(resumed, '_batch', wraps=resumed._batch) as calls:
            self.f.quiet_run(resumed)
        self.assertEqual([call.args[1] for call in calls.call_args_list], [1, 2])
        ck = load_checkpoint(run.run_dir / 'last.ckpt', run.identity)
        self.assertFalse(ck['progress']['pending_validation'])
        self.assertEqual(len(ck['development_state']['validation_history']), 1)
        self.assertTrue((run.run_dir / 'best-dev.ckpt').exists())

    def test_controlled_epoch_extension_and_strict_identity(self):
        run = self.trainer(self.root / 'extended', epochs=1)
        self.f.quiet_run(run)
        extended = self.trainer(run.run_dir, resume=True, epochs=1, extend_to=6)
        self.f.quiet_run(extended)
        ck = load_checkpoint(run.run_dir / 'last.ckpt', run.identity)
        self.assertEqual(ck['progress']['epoch'], 2)
        self.assertEqual(ck['horizon']['original_total_steps'], 3)
        self.assertEqual(ck['horizon']['total_steps'], 6)
        self.assertEqual(ck['identity']['training']['max_epochs'], 1)
        third = self.trainer(run.run_dir, resume=True, epochs=1, extend_to=9)
        self.f.quiet_run(third)
        self.assertEqual(third.state['epoch'], 3)
        for key, value in (('loss', {}), ('data', {'split_hash': 'changed'}), ('training', {'seed': 3})):
            changed = copy.deepcopy(run.identity); changed[key] = value
            with self.assertRaises(ValueError): load_checkpoint(run.run_dir / 'last.ckpt', changed, extension=True)
        with self.assertRaises(ValueError): self.trainer(run.run_dir, resume=True, epochs=1, extend_to=10)

    def test_best_dev_strict_improvement_and_all_validation_cases(self):
        run = self.trainer(self.root / 'best', epochs=30)
        run.validation_case_ids = [f'fake_dev_{i}' for i in range(40)]
        class FortyCases(fixtures.RandomCases):
            def __len__(self): return 40
            def __getitem__(self, index): return super().__getitem__(index % 3)
        run.validation_dataset = FortyCases()
        scores = {30: 0.8, 60: 0.7, 90: 0.8}  # 第 10/20/30 个 epoch 的验证分数
        def controlled_score(records):
            result = summarize_regions(records)
            self.assertEqual(len(records), 40)
            result['mean_case_dice'] = scores[run.state['global_step']]
            return result
        with patch('organ_relation.training.brain_tumor.summarize_regions', side_effect=controlled_score):
            self.f.quiet_run(run)
        best = load_checkpoint(run.run_dir / 'best-dev.ckpt', run.identity)
        last = load_checkpoint(run.run_dir / 'last.ckpt', run.identity)
        self.assertEqual(best['progress']['epoch'], 10)
        self.assertEqual(last['progress']['epoch'], 30)
        self.assertEqual([h['improved'] for h in last['development_state']['validation_history']], [True, False, False])
        vals = [r for r in self.f.rows(run.run_dir) if r['phase'] == 'internal_dev']
        self.assertEqual([len(r['region_diagnostic_cases']) for r in vals], [40, 40, 40])

    def test_300_400_500_epoch_extension_budget(self):
        opt = dict(max_steps=None, max_epochs=300, cadence_unit='epoch')
        identity = dict(provenance='synthetic', training=opt)
        ck = dict(identity=identity, origin_identity=identity, progress={'global_step': 48000})
        parent = self.root / 'synthetic_parent'; parent.write_bytes(b'synthetic test')
        ck['horizon'] = resolve_horizon(opt, 160, None, None, identity, None)
        ck['horizon'] = resolve_horizon(opt, 160, ck, 64000, identity, parent)
        self.assertEqual(ck['horizon']['total_steps'], 64000)
        ck['progress']['global_step'] = 64000
        ck['horizon'] = resolve_horizon(opt, 160, ck, 80000, identity, parent)
        self.assertEqual(ck['horizon']['total_steps'], 80000)


if __name__ == '__main__':
    unittest.main()
