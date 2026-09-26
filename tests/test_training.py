"""CPU acceptance for transactional training and independent case recovery."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch import nn
from organ_relation.data.full_scan import FullScanSample
from organ_relation.data.splits import (create_development, load_split, select_cases,
                                       training_manifest, validate_artifact, write_split)
from organ_relation.evaluation.progress import CaseLedger
from organ_relation.losses import JointLoss
from organ_relation.metrics import hard_dice, summarize_dice
from organ_relation.models.segmentor import SegmentorOutput
from organ_relation.training.engine import Trainer, validate_options
from organ_relation.training.progress import ProgressClock
from organ_relation.training.state import (ScalarLog, atomic_bytes, atomic_json, capture_rng,
    digest, load_checkpoint, save_checkpoint, seed_all)
from test_full_scan import synthetic_pair


def config():
    return json.loads((ROOT / 'configs/train_sanity.json').read_text())


class RandomCases(torch.utils.data.Dataset):
    """Variable full volumes consume all three CPU RNGs to expose bad restore."""
    def __len__(self):
        return 3

    def __getitem__(self, index):
        shape = (2 + index, 3, 4)
        image = torch.rand(1, *shape) + random.random() + float(np.random.random())
        label = torch.arange(np.prod(shape)).reshape(shape).long() % 16
        return FullScanSample(image, label, {})


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv3d(1, 3, 1)
        self.coarse_head = nn.Conv3d(3, 16, 1)
        self.decoder = nn.Conv3d(3, 16, 1)

    def forward(self, image):
        feature = torch.tanh(self.encoder(image))
        return SegmentorOutput(self.coarse_head(feature), self.decoder(feature))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def trainer(self, directory, *, resume=False, validation=False, options=None, identity=None):
        seed_all(712)
        model = TinyModel().to(memory_format=torch.channels_last_3d)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0, foreach=False, fused=False)
        opts = config()['training']
        opts.update(max_steps=7, validation_every=2 if validation else 0)
        if options:
            opts.update(options)
        ident = identity or dict(training=opts, model='tiny', optimizer='AdamW',
                                 preprocessing='synthetic', provenance='test', data={'manifest_hash': 'test'})
        dataset = RandomCases()
        return Trainer(model, JointLoss(epsilon=1e-6, lambda_c=.5, align_corners=False),
            optimizer, dataset, ['one', 'two', 'three'], device='cpu', options=opts,
            identity=ident, run_dir=directory, validation_dataset=dataset if validation else None,
            validation_case_ids=['one', 'two', 'three'] if validation else [],
            resume=directory / 'last.ckpt' if resume else None)

    def quiet_run(self, trainer, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return trainer.run(**kwargs)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def rows(self, directory):
        return [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]

    def compare_resume(self, k, validation=False):
        whole = self.trainer(self.root / 'whole', validation=validation)
        self.quiet_run(whole)
        expected = load_checkpoint(whole.run_dir / 'last.ckpt', whole.identity)
        split = self.trainer(self.root / 'split', validation=validation)
        self.quiet_run(split, stop_after=k)
        middle = load_checkpoint(split.run_dir / 'last.ckpt', split.identity)
        self.assertEqual(middle['progress']['global_step'], k)
        self.assertEqual(middle['progress']['cursor'], k % 3)
        if k % 3:
            self.assertEqual(sorted(middle['progress']['order']), [0, 1, 2])
        # Deliberately perturb all RNGs before reconstruction.
        random.random(); np.random.random(); torch.rand(3)
        resumed = self.trainer(split.run_dir, resume=True, validation=validation)
        self.assertEqual(len(resumed.clock.samples), 0)
        self.quiet_run(resumed)
        actual = load_checkpoint(split.run_dir / 'last.ckpt', resumed.identity)
        for key in ('model', 'optimizer', 'progress', 'rng', 'sampler_generator', 'loader_generator'):
            self.assert_nested_equal(expected[key], actual[key])
        a, b = self.rows(whole.run_dir), self.rows(split.run_dir)
        for rows in (a, b):
            self.assertEqual([r['global_step'] for r in rows if r['phase'] == 'train'], list(range(1, 8)))
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            for key in ('phase', 'epoch', 'global_step', 'case_id', 'total_loss', 'coarse', 'final', 'metrics', 'diagnostics'):
                self.assertEqual(x.get(key), y.get(key), key)
        last = b[-1]['progress']
        self.assertEqual(last['eta_seconds'], 0)
        self.assertEqual(last['epoch_eta_seconds'], 0)

    def test_exact_mid_epoch_resume_all_state_and_trajectory(self):
        self.compare_resume(2)

    def test_exact_epoch_boundary_resume_all_state_and_trajectory(self):
        self.compare_resume(3)

    def test_resume_with_validation_keeps_training_rng_and_trajectory(self):
        self.compare_resume(2, validation=True)

    def test_pending_validation_skips_completed_case_after_interruption(self):
        trainer = self.trainer(self.root / 'run', validation=True)
        original = CaseLedger.commit
        def interrupt(ledger, case, data, result):
            if case == 'two':
                raise OSError('simulated interruption')
            return original(ledger, case, data, result)
        with patch.object(CaseLedger, 'commit', interrupt), self.assertRaises(OSError):
            self.quiet_run(trainer)
        checkpoint = load_checkpoint(trainer.run_dir / 'last.ckpt', trainer.identity)
        self.assertEqual(checkpoint['progress']['global_step'], 2)
        self.assertTrue(checkpoint['progress']['pending_validation'])
        resumed = self.trainer(trainer.run_dir, resume=True, validation=True)
        written = []
        def track(ledger, case, data, result):
            written.append(case)
            return original(ledger, case, data, result)
        with patch.object(CaseLedger, 'commit', track):
            self.quiet_run(resumed, stop_after=1)
        self.assertEqual(written, ['two', 'three'])
        rows = self.rows(trainer.run_dir)
        self.assertEqual(sum(r['phase'] == 'train_monitor' for r in rows), 1)
        self.assertFalse(resumed.state['pending_validation'])

    def test_tail_ahead_of_checkpoint_is_removed_without_duplicate_steps(self):
        trainer = self.trainer(self.root / 'run')
        self.quiet_run(trainer, stop_after=2)
        committed = (trainer.run_dir / 'metrics.jsonl').read_bytes()
        trainer.log.append({'global_step': 3, 'uncommitted': True})
        with contextlib.redirect_stdout(io.StringIO()):
            resumed = self.trainer(trainer.run_dir, resume=True)
        self.assertEqual((trainer.run_dir / 'metrics.jsonl').read_bytes(), committed)
        self.quiet_run(resumed)
        self.assertEqual([r['global_step'] for r in self.rows(trainer.run_dir)], list(range(1, 8)))

    def test_changed_resume_identity_fields_rejected(self):
        trainer = self.trainer(self.root / 'run')
        self.quiet_run(trainer, stop_after=1)
        for key in ('model', 'optimizer', 'preprocessing', 'provenance', 'data', 'training'):
            identity = copy.deepcopy(trainer.identity); identity[key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                load_checkpoint(trainer.run_dir / 'last.ckpt', identity)

    def test_incomplete_corrupt_and_invalid_progress_checkpoints_rejected(self):
        trainer = self.trainer(self.root / 'run')
        self.quiet_run(trainer, stop_after=1)
        path = trainer.run_dir / 'last.ckpt'; good = path.read_bytes()
        for bad in (b'', good[:20], good[:-1], good[:-1] + bytes([good[-1] ^ 1])):
            path.write_bytes(bad)
            with self.assertRaises(ValueError):
                load_checkpoint(path, trainer.identity)
        path.write_bytes(good)
        state = load_checkpoint(path, trainer.identity)
        incomplete = dict(state); del incomplete['rng']
        save_checkpoint(path, incomplete)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            load_checkpoint(path, trainer.identity)
        state['progress']['cursor'] = 0
        save_checkpoint(path, state)
        with self.assertRaisesRegex(ValueError, 'progress'):
            self.trainer(trainer.run_dir, resume=True)

    def test_failed_optimizer_step_never_commits_half_step(self):
        trainer = self.trainer(self.root / 'run')
        self.quiet_run(trainer, stop_after=1)
        ckpt = (trainer.run_dir / 'last.ckpt').read_bytes()
        log = (trainer.run_dir / 'metrics.jsonl').read_bytes()
        def broken():
            with torch.no_grad():
                next(trainer.model.parameters()).add_(1)
            raise RuntimeError('half update')
        with patch.object(trainer.optimizer, 'step', broken), self.assertRaises(RuntimeError):
            self.quiet_run(trainer)
        self.assertEqual((trainer.run_dir / 'last.ckpt').read_bytes(), ckpt)
        self.assertEqual((trainer.run_dir / 'metrics.jsonl').read_bytes(), log)
        resumed = self.trainer(trainer.run_dir, resume=True)
        self.assertEqual(resumed.state['global_step'], 1)

    def test_checkpoint_and_diagnostics_frequencies_are_configurable(self):
        trainer = self.trainer(self.root / 'run', options={'checkpoint_every': 5, 'diagnostics_every': 2})
        with patch.object(trainer, 'checkpoint', wraps=trainer.checkpoint) as save:
            self.quiet_run(trainer)
        self.assertEqual(save.call_count, 2)  # step 5 and final step 7
        rows = self.rows(trainer.run_dir)
        for row in rows:
            self.assertEqual(row['diagnostics'] is not None, row['global_step'] % 2 == 0)
            if row['diagnostics']:
                for module in row['diagnostics'].values():
                    self.assertTrue(module['finite'])
                    self.assertGreater(module['gradient_norm'], 0)
                    self.assertGreater(module['update_norm'], 0)

    def test_wrong_run_and_existing_new_run_rejected(self):
        trainer = self.trainer(self.root / 'run')
        self.quiet_run(trainer, stop_after=1)
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.trainer(trainer.run_dir)
        run = json.loads((trainer.run_dir / 'run.json').read_text()); run['run_id'] = 'wrong'
        atomic_json(trainer.run_dir / 'run.json', run)
        with self.assertRaisesRegex(ValueError, 'original run identity'):
            self.trainer(trainer.run_dir, resume=True)

    def test_options_reject_unsupported_execution(self):
        for key, value in [('batch_size', 2), ('num_workers', 1), ('scheduler', 'cosine'),
                           ('checkpoint_every', 0), ('diagnostics_every', -1), ('max_steps', 0)]:
            options = config()['training']; options[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_options(options)

    def test_atomic_replace_failure_preserves_previous_file(self):
        path = self.root / 'state'; path.write_bytes(b'old')
        with patch('organ_relation.training.state.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                atomic_bytes(path, b'new')
        self.assertEqual(path.read_bytes(), b'old')
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_scalar_log_rejects_tensor_nan_and_damaged_committed_prefix(self):
        log = ScalarLog(self.root / 'log')
        for value in (torch.tensor(1.), float('nan')):
            with self.assertRaises((TypeError, ValueError)):
                log.append({'value': value})
        log.append({'value': 1}); position = log.position(1)
        log.path.write_bytes(b'x' * position['bytes'])
        with self.assertRaisesRegex(ValueError, 'checksum'):
            log.recover(position)
        log.path.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'shorter'):
            log.recover(position)

    def test_eta_warmup_rolling_window_separate_clocks_and_completion(self):
        train, validation = ProgressClock(), ProgressClock()
        for t in range(1, 5):
            train.add(t)
            self.assertEqual(train.estimate(7)['status'], 'warming up')
        train.add(5)
        self.assertEqual(train.estimate(7, 2)['eta_seconds'], 21)
        self.assertEqual(train.estimate(7, 2)['epoch_eta_seconds'], 6)
        for t in range(6, 26):
            train.add(t)
        self.assertEqual(train.estimate(2)['rolling_seconds'], 15.5)
        self.assertEqual(validation.estimate(2)['status'], 'warming up')
        self.assertEqual(train.estimate(0, 0)['eta_seconds'], 0)
        self.assertEqual(ProgressClock().estimate(0)['status'], 'complete')

    def test_hard_dice_empty_and_false_positive_rules(self):
        gt = torch.tensor([1, 1, 2, 0, 0]).reshape(1, 1, 5)
        pred = torch.tensor([1, 0, 0, 3, 0]).reshape_as(gt)
        result = hard_dice(pred, gt)
        self.assertAlmostEqual(result['organs'][0]['dice'], 2/3)
        self.assertEqual(result['organs'][1]['dice'], 0)
        self.assertEqual(result['organs'][2]['dice'], 0)
        self.assertEqual(result['organs'][2]['false_positive_voxels'], 1)
        self.assertIsNone(result['organs'][3]['dice'])
        self.assertEqual(len(result['organs']), 15)
        self.assertAlmostEqual(result['mean_dice'], 2/9)

    def test_metrics_are_per_case_not_volume_weighted_and_no_test_gt_assumption(self):
        one = torch.ones(1, 1, 1, dtype=torch.int64)
        many = torch.ones(1, 1, 100, dtype=torch.int64)
        summary = summarize_dice([hard_dice(one, one), hard_dice(torch.zeros_like(many), many)])
        self.assertEqual(summary['mean_case_dice'], .5)
        self.assertEqual(summary['organs'][0]['valid_cases'], 2)
        self.assertEqual(summary['organs'][1]['valid_cases'], 0)
        self.assertIsNone(summary['nsd'])
        with self.assertRaisesRegex(ValueError, 'requires a label'):
            hard_dice(one, None)
        for bad in (one.float(), one * 16, one[0]):
            with self.assertRaises(ValueError):
                hard_dice(bad, one)

    def test_case_ledger_valid_missing_corrupt_and_changed_data(self):
        ledger = CaseLedger(self.root / 'eval', {'model': 'a', 'preprocessing': 'B', 'manifest': 'x'})
        ledger.commit('one', {'header': 'v1'}, {'dice': .5})
        reopened = CaseLedger(ledger.directory, ledger.identity)
        self.assertEqual(reopened.read('one', {'header': 'v1'}), {'dice': .5})
        self.assertIsNone(reopened.read('one', {'header': 'v2'}))
        path = ledger._path('one'); path.write_text('broken')
        self.assertIsNone(reopened.read('one', {'header': 'v1'}))
        path.unlink()
        self.assertIsNone(reopened.read('one', {'header': 'v1'}))

    def test_case_ledger_config_weight_manifest_mismatch_rejected(self):
        identity = dict(model='a', preprocessing='B', manifest='x', metric='dice', run='1')
        ledger = CaseLedger(self.root / 'eval', identity); ledger.commit('one', {}, {'dice': 1})
        for key in identity:
            changed = dict(identity); changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'identity mismatch'):
                CaseLedger(ledger.directory, changed)

    def test_result_write_without_ledger_commit_is_not_skipped(self):
        ledger = CaseLedger(self.root / 'eval', {'model': 'a'})
        real_write = atomic_json
        def fail_ledger(path, value):
            if Path(path).name == 'ledger.json':
                raise OSError('power loss')
            real_write(path, value)
        with patch('organ_relation.evaluation.progress.atomic_json', fail_ledger), self.assertRaises(OSError):
            ledger.commit('test-image-only', {'label': None}, {'prediction': [1, 2, 0]})
        self.assertTrue(ledger._path('test-image-only').exists())
        self.assertIsNone(ledger.read('test-image-only', {'label': None}))
        restarted = CaseLedger(ledger.directory, ledger.identity)
        self.assertIsNone(restarted.read('test-image-only', {'label': None}))
        with patch('organ_relation.metrics.hard_dice', side_effect=AssertionError('no GT')):
            restarted.commit('test-image-only', {'label': None}, {'prediction': [1, 2, 0]})
            self.assertEqual(restarted.read('test-image-only', {'label': None})['prediction'], [1, 2, 0])

    def manifest(self):
        records = [dict(case_id=f'amos_{i:04}', image=f'imagesTr/amos_{i:04}.nii.gz',
            label=f'labelsTr/amos_{i:04}.nii.gz', original_filename=f'amos_{i:04}.nii.gz',
            source='official_training', modality='CT', verified=True) for i in range(1, 201)]
        for source, case, folder, label in [('official_validation', 'va', 'imagesVa', 'labelsVa/va.nii.gz'),
                                           ('official_test', 'ts', 'imagesTs', None)]:
            records.append(dict(case_id=case, image=f'{folder}/{case}.nii.gz', label=label,
                original_filename=f'{case}.nii.gz', source=source, modality='CT', verified=True))
        return {'records': records, 'source_manifest_sha256': 'test'}

    def test_frozen_split_repeatability_hash_and_no_overwrite(self):
        before = capture_rng(torch.device('cpu'))
        a = create_development(self.manifest(), 17, 160)
        self.assert_nested_equal(before, capture_rng(torch.device('cpu')))
        self.assertEqual(a, create_development(self.manifest(), 17, 160))
        self.assertEqual((len(a['development']['train']), len(a['development']['internal_dev'])), (160, 40))
        path = self.root / 'split.json'; write_split(path, a)
        self.assertEqual(load_split(path), a)
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            write_split(path, a)
        a['development']['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'hash'):
            validate_artifact(a)

    def test_roles_prevent_leakage_and_final_requires_full_pool(self):
        artifact = create_development(self.manifest(), 17, 160)
        self.assertEqual(len(select_cases(artifact, 'final_train')), 200)
        self.assertEqual(select_cases(artifact, 'official_test')[0]['label'], None)
        self.assertEqual(select_cases(artifact, 'official_validation')[0]['case_id'], 'va')
        for case in ('va', 'ts', artifact['development']['internal_dev'][0]):
            with self.assertRaisesRegex(ValueError, 'violates role'):
                select_cases(artifact, 'development_train', [case])
        with self.assertRaisesRegex(ValueError, 'complete official'):
            select_cases(artifact, 'final_train', limit=2)

    def test_split_rejects_overlap_unverified_test_labels_and_absolute_paths(self):
        artifact = create_development(self.manifest(), 17, 160)
        for key, value in [('verified', False), ('image', 'C:/data/image.nii.gz'), ('modality', 'MRI')]:
            bad = copy.deepcopy(artifact); bad['manifest']['records'][0][key] = value
            with self.assertRaises(ValueError):
                validate_artifact(bad)
        bad = copy.deepcopy(artifact); bad['manifest']['records'][-1]['label'] = 'fake.nii.gz'
        with self.assertRaisesRegex(ValueError, 'image-only'):
            validate_artifact(bad)
        bad = copy.deepcopy(artifact); bad['development']['internal_dev'][0] = bad['development']['train'][0]
        with self.assertRaisesRegex(ValueError, 'partition'):
            validate_artifact(bad)

    def test_cli_synthetic_nifti_complete_segmentor_loss_update_and_resume(self):
        spec = importlib.util.spec_from_file_location('training_cli', ROOT / 'scripts/train.py')
        cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
        data = self.root / 'data'; data.mkdir(); synthetic_pair(data, shape=(12, 12, 12))
        for folder in ('imagesTr', 'labelsTr'):
            shutil.copyfile(data / folder / 'amos_0001.nii.gz', data / folder / 'amos_0002.nii.gz')
        atomic_json(data / 'dataset.json', {'training': [dict(image=f'imagesTr/amos_{i:04}.nii.gz',
                    label=f'labelsTr/amos_{i:04}.nii.gz') for i in (1, 2)]})
        selection = json.loads((ROOT / 'configs/ct_stats.json').read_text())
        artifact = create_development(training_manifest(data, selection), 17, 1)
        split_path = self.root / 'split.json'; write_split(split_path, artifact)
        cfg = config()
        for key, name in [('model_config', 'segmentor_micro.json'), ('baseline_config', 'baseline.json'),
                          ('selection_config', 'ct_stats.json')]:
            cfg[key] = str(ROOT / 'configs' / name)
        cfg['runtime'].update(backend='cpu', device='cpu', cpu_threads=1)
        cfg['training'].update(max_steps=2, validation_every=1)
        cfg_path = self.root / 'config.json'; atomic_json(cfg_path, cfg)
        run = self.root / 'run'
        argv = ['--config', str(cfg_path), '--data-root', str(data), '--split', str(split_path),
                '--run-dir', str(run), '--cpu-synthetic']
        originals = {p: p.read_bytes() for p in data.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv + ['--stop-after', '1']), 0)
            self.assertEqual(cli.main(argv + ['--resume', str(run / 'last.ckpt')]), 0)
        rows = [r for r in self.rows(run) if r['phase'] == 'train']
        self.assertEqual([r['global_step'] for r in rows], [1, 2])
        self.assertEqual(rows[0]['shape'], [12, 9, 9])
        for row in rows:
            self.assertEqual(set(row['diagnostics']), {'encoder', 'coarse_head', 'relation', 'node_to_space', 'fusion', 'decoder'})
            for name, stats in row['diagnostics'].items():
                self.assertTrue(stats['finite'], name)
                self.assertGreater(stats['gradient_norm'], 0, name)
                self.assertGreater(stats['update_norm'], 0, name)
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw)
        with patch.object(cli, 'git_state', return_value={'commit': None}), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(argv), 2)


if __name__ == '__main__':
    unittest.main()
