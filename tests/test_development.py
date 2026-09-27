"""Synthetic-only acceptance of frozen development orchestration, not a new split."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from organ_relation.models.monai_reference import ReferenceOutput
from organ_relation.training.development import (verify_frozen_split, summarize_branches,
                                                cadence_due, development_scalars)
from organ_relation.training.engine import Trainer
from organ_relation.training.monai_reference import MonaiReferenceLoss
from organ_relation.training.state import atomic_json, digest, load_checkpoint, seed_all
from organ_relation.training.tensorboard import scalar_values
from organ_relation.training.extension import resolve_horizon
from organ_relation.data.splits import load_split

ROOT = fixtures.ROOT
HASH = '7d308eca4f7324f0e899c7416a45a03f8dfbec5e867e5ed6018353e96541468c'


def config(name='reference'):
    return json.loads((ROOT / f'configs/train_monai_{name}_A_160_40.json').read_text())


class TinyReference(nn.Module):
    """Exercise orchestration cheaply; real MONAI model coverage stays in its suite."""
    def __init__(self):
        super().__init__()
        self.network = nn.Conv3d(1, 16, 1)

    def forward(self, image):
        return ReferenceOutput(self.network(image))


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def trainer(self, path, resume=False, extend_to=None, epochs=20):
        seed_all(20260925)
        cfg = config()
        options = cfg['training']
        options.update(max_epochs=epochs, memory_format='contiguous')
        model = TinyReference()
        criterion = MonaiReferenceLoss(cfg['loss'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        ident = dict(training=options, mode=cfg['mode'], model='tiny', loss=cfg['loss'],
                     optimizer=cfg['optimizer'], preprocessing='synthetic', provenance='test',
                     data={'manifest_hash': 'synthetic', 'split_hash': 'synthetic'})
        cases = fixtures.RandomCases()
        return Trainer(model, criterion, optimizer, cases, ['one', 'two', 'three'],
            device='cpu', options=options, identity=ident, run_dir=path,
            validation_dataset=cases, validation_case_ids=['one', 'two', 'three'],
            resume=path / 'last.ckpt' if resume else None, extend_to=extend_to, tensorboard=True)

    def test_configs_equal_comparison_budget_and_unchanged_backbone_loss(self):
        a, c = config(), config('relation')
        for key in ('data', 'training', 'optimizer', 'runtime', 'model', 'loss', 'input_processing',
                    'split_artifact', 'expected_split_hash'):
            self.assertEqual(a[key], c[key], key)
        self.assertEqual(a['expected_split_hash'], HASH)
        self.assertEqual(a['split_artifact'], '../runs/splits/development_160_40.json')
        self.assertEqual(a['data']['candidate'], 'A')
        self.assertEqual(a['data']['role'], 'development_train')
        self.assertTrue(all(a['data'][k] is None for k in ('train_cases', 'train_limit', 'validation_cases', 'validation_limit')))
        self.assertEqual(a['training']['max_epochs'], 300)
        self.assertIsNone(a['training']['max_steps'])
        self.assertIsNone(a['training']['scheduler'])
        self.assertEqual(a['training']['seed'], 20260925)
        self.assertEqual(resolve_horizon(a['training'], 160, None, None, {'provenance': 'test'}, None)['total_steps'], 48000)
        for name, current in (('reference', a), ('relation', c)):
            previous = json.loads((ROOT / f'configs/train_monai_{name}_whole_volume_overfit.json').read_text())
            for key in ('model', 'loss', 'input_processing', 'optimizer', 'runtime', 'relation', 'coarse_supervision'):
                self.assertEqual(current.get(key), previous.get(key))
        baseline = json.loads((ROOT / 'configs/baseline.json').read_text())
        self.assertEqual(baseline['preprocessing']['spacing_candidates']['A'], [1.5, 1.5, 3.0])

    def test_frozen_hash_internal_not_file_checksum_counts_tampering(self):
        # Explicit fake IDs and temporary file, never manufacture the real split.
        records = [dict(case_id=f'synthetic_{i}', source='official_training', modality='CT', verified=True,
                    image=f'images/{i}.nii', label=f'labels/{i}.nii', original_filename=f'{i}.nii') for i in range(200)]
        manifest = dict(records=records)
        split = dict(seed=99, train=[r['case_id'] for r in records[:160]], internal_dev=[r['case_id'] for r in records[160:]])
        artifact = dict(schema_version=1, manifest=manifest, development=split,
                        manifest_hash=digest(manifest), split_hash=digest(split))
        path = self.root / 'synthetic_only.json'
        atomic_json(path, artifact)
        a = load_split(path)
        result = verify_frozen_split(path, a, artifact['split_hash'])
        self.assertNotEqual(result['file_sha256'], result['split_hash'])
        for bad_hash in (HASH, result['file_sha256']):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify_frozen_split(path, a, bad_hash)
        a['development']['train'].pop()
        with self.assertRaisesRegex(ValueError, '160/40'):
            verify_frozen_split(path, a, a['split_hash'])
        atomic_json(path, a)
        with self.assertRaises(ValueError): load_split(path)

    def test_cadence_exact_epochs_and_no_step25_monitor(self):
        opt = config()['training']
        vals, checkpoints = [], []
        for step in range(1, 48001):
            epoch = (step - 1) // 160 + 1
            end = step % 160 == 0
            if cadence_due(opt, 'validation_every', step, epoch, end): vals.append(epoch)
            if cadence_due(opt, 'checkpoint_every', step, epoch, end): checkpoints.append(step)
        self.assertEqual(vals, list(range(10, 301, 10)))
        self.assertEqual(checkpoints, list(range(160, 48001, 160)))

    def test_epochs_resume_best_history_jsonl_and_board_whitelist(self):
        whole = self.trainer(self.root / 'whole')
        self.f.quiet_run(whole)
        first = self.trainer(self.root / 'resumed')
        self.f.quiet_run(first, stop_after=29)  # Partial epoch immediately before validation.
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
        self.assertTrue(any(r['diagnostics'] for r in train))
        self.assertIn('memory', train[0])
        self.assertIn('gt_foreground_true_class_mean_probability', vals[0]['diagnostic_cases']['one'])
        self.assertAlmostEqual(epochs[0]['epoch_metrics']['total_loss'], sum(r['total_loss'] for r in train[:3]) / 3)
        self.assertEqual(scalar_values(train[0]), {})
        events = EventAccumulator(str(second.board.directory), size_guidance={'scalars': 0}).Reload()
        expected = {'Loss/Train_Total', 'Loss/Train_Final', 'Loss/Val_Final', 'Dice/Train_Final_Hard',
            'Dice/Train_Final_Soft', 'Dice/Val_Final_Hard', 'Dice/Val_Final_Soft'}
        expected |= {f'Dice_Per_Class_Val_Final/Class_{i:02d}' for i in range(1, 16)}
        self.assertEqual(set(events.Tags()['scalars']), expected)
        self.assertEqual([e.step for e in events.Scalars('Loss/Train_Total')], list(range(3, 61, 3)))
        self.assertEqual([e.step for e in events.Scalars('Dice/Val_Final_Hard')], [30, 60])

    def test_pending_validation_ledger_resume_without_retraining(self):
        run = self.trainer(self.root / 'pending', epochs=10)
        original = run._batch
        visited = []
        def interrupt(dataset, index, *, validation=False):
            if validation:
                visited.append(index)
                if index == 1: raise RuntimeError('synthetic interruption')
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

    def test_c_coarse_whitelist_and_absent_hard_class_preserved(self):
        hard = fixtures.hard_dice(torch.ones(2,3,4,dtype=torch.long), torch.ones(2,3,4,dtype=torch.long))
        branch = dict(loss=2.0, hard=hard, soft_per_organ=[0.25]*15)
        case = dict(total_loss=3., final=branch, coarse=branch)
        summary = summarize_branches([case, case])
        self.assertEqual(summary['final']['hard']['organs'][1]['mean_dice'], None)
        val = development_scalars(dict(phase='internal_dev', epoch_metrics=summary))
        self.assertEqual(val['Dice/Val_Coarse_Hard'], 1.)
        self.assertEqual(val['Loss/Val_Coarse'], 2.)
        self.assertEqual(val['Dice_Per_Class_Val_Coarse/Class_01'], 1.)
        self.assertNotIn('Dice_Per_Class_Val_Coarse/Class_02', val)
        train = development_scalars(dict(phase='train_epoch', epoch_metrics=summary))
        self.assertEqual(train['Loss/Train_Total'], 3.)
        self.assertEqual(train['Dice/Train_Coarse_Soft'], .25)
        self.assertFalse(any('Per_Class' in k for k in train))

    def test_actual_c_epoch_metrics_and_resume(self):
        from test_monai_relation import MonaiRelationTests
        fixture = MonaiRelationTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.trainer
        # Its existing real MONAI fixture has two synthetic cases. Configure an
        # epoch run before Trainer construction rather than mutating identity.
        base_config = fixture.proposed_config
        def epoch_config():
            c = base_config()
            c['training']['cadence_unit'] = 'epoch'
            return c
        with patch.object(fixture, 'proposed_config', side_effect=epoch_config):
            whole = original(fixture.fixture.root / 'epoch_c')
            self.f.quiet_run(whole)
            first = original(fixture.fixture.root / 'epoch_c_resumed')
            self.f.quiet_run(first, stop_after=1)
            second = original(first.run_dir, resume=True)
            self.f.quiet_run(second)
        a = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        b = load_checkpoint(second.run_dir / 'last.ckpt', second.identity)
        for key in ('model', 'optimizer', 'progress', 'rng', 'development_state'):
            self.f.assert_nested_equal(a[key], b[key])
        rows = self.f.rows(whole.run_dir)
        summaries = [r for r in rows if r['phase'] == 'train_epoch']
        self.assertTrue(summaries)
        self.assertIn('coarse', summaries[0]['epoch_metrics'])
        self.assertIn('Loss/Train_Coarse', scalar_values(summaries[0]))
        self.assertIn('best_dev', load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)['development_state'])

    def test_best_dev_strict_improvement_and_all_validation_cases(self):
        run = self.trainer(self.root / 'best', epochs=30)
        run.validation_case_ids = [f'fake_dev_{i}' for i in range(40)]
        class FortyCases(fixtures.RandomCases):
            def __len__(self): return 40
            def __getitem__(self, index): return super().__getitem__(index % 3)
        run.validation_dataset = FortyCases()
        from organ_relation.metrics import summarize_dice
        scores = iter((0.8, 0.7, 0.8))
        def controlled_score(records):
            result = summarize_dice(records)
            self.assertEqual(len(records), 40)
            result['mean_case_dice'] = next(scores)
            return result
        with patch('organ_relation.training.engine.summarize_dice', side_effect=controlled_score):
            self.f.quiet_run(run)
        best = load_checkpoint(run.run_dir / 'best-dev.ckpt', run.identity)
        last = load_checkpoint(run.run_dir / 'last.ckpt', run.identity)
        self.assertEqual(best['progress']['epoch'], 10)
        self.assertEqual(last['progress']['epoch'], 30)
        self.assertEqual([h['improved'] for h in last['development_state']['validation_history']], [True, False, False])
        vals = [r for r in self.f.rows(run.run_dir) if r['phase'] == 'internal_dev']
        self.assertEqual([len(r['diagnostic_cases']) for r in vals], [40, 40, 40])

    def test_300_400_500_epoch_extension_budget(self):
        opt = config()['training']
        identity = dict(provenance='synthetic', training=opt)
        ck = dict(identity=identity, origin_identity=identity, progress={'global_step': 48000})
        parent = self.root / 'synthetic_parent'; parent.write_bytes(b'synthetic test')
        ck['horizon'] = resolve_horizon(opt, 160, None, None, identity, None)
        ck['horizon'] = resolve_horizon(opt, 160, ck, 64000, identity, parent)
        self.assertEqual(ck['horizon']['total_steps'], 64000)
        ck['progress']['global_step'] = 64000
        ck['horizon'] = resolve_horizon(opt, 160, ck, 80000, identity, parent)
        self.assertEqual(ck['horizon']['total_steps'], 80000)
        self.assertEqual(opt['max_epochs'], 300)

    def test_formal_cli_refuses_subset_and_split_generation_before_data(self):
        spec = importlib.util.spec_from_file_location('development_cli', ROOT / 'scripts/train.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        argv = ['--config', str(ROOT / 'configs/train_monai_reference_A_160_40.json'), '--data-root', str(self.root)]
        for extra in (['--cases', 'amos_0109'], ['--prepare-split', str(self.root / 'forbidden.json')], ['--extend-epochs', '400']):
            with contextlib.redirect_stderr(io.StringIO()): self.assertEqual(cli.main(argv + extra), 2)
        self.assertFalse((self.root / 'forbidden.json').exists())


if __name__ == '__main__': unittest.main()
