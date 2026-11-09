"""--auto-resume: one command that starts, resumes, or restarts a run interrupted before its first checkpoint."""
import contextlib
import importlib.util
import io
import json
import unittest
from datetime import datetime, timezone

import torch

import test_training as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer
from organ_relation.training.state import load_checkpoint, prepare_auto_resume, seed_all

NOW = datetime(2026, 11, 9, 4, 30, tzinfo=timezone.utc)


class AutoResumeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def identity(self, **changes):
        options = dict(self.cfg['training'], max_steps=2, validation_every=0, diagnostics_every=1,
                       checkpoint_every=1, validation_role='train_monitor')
        identity = dict(mode='brats_regions', model=self.cfg['model'], loss=self.cfg['loss'], threshold=.5,
                        training=options, preprocessing=self.cfg['preprocessing'], provenance='test',
                        data={'manifest_hash': 'synthetic'})
        identity.update(changes)
        return identity

    def trainer(self, path, identity, resume=None):
        seed_all(171)
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']))
        return Trainer(model, RegionJointLoss(**self.cfg['loss']),
                       torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False),
                       Cases(), ['one', 'two'], device='cpu', options=identity['training'], identity=identity,
                       run_dir=path, resume=resume, tensorboard=False)

    def test_missing_or_empty_directory_starts_a_new_run(self):
        path = self.fixture.root / 'new'
        self.assertEqual(prepare_auto_resume(path, self.identity()), (None, None))
        path.mkdir()
        self.assertEqual(prepare_auto_resume(path, self.identity()), (None, None))
        self.assertTrue(path.is_dir())

    def test_interrupted_before_first_checkpoint_is_archived_and_restarts(self):
        path = self.fixture.root / 'run_fold0'
        identity = self.identity()
        self.trainer(path, identity)  # Writes run.json and the log, then "Ctrl+C" before any checkpoint.
        (path / 'tensorboard').mkdir()
        (path / 'tensorboard' / 'events.out').write_text('x')
        before = sorted(p.relative_to(path).as_posix() for p in path.rglob('*'))
        with self.assertRaisesRegex(ValueError, 'must be empty'):
            self.trainer(path, identity)  # The plain command is what used to get stuck.
        later = self.identity(provenance='another commit')  # Code version may differ; the experiment may not.
        resume, archived = prepare_auto_resume(path, later, now=NOW)
        self.assertIsNone(resume)
        self.assertEqual(archived.name, 'run_fold0.aborted-20261109T043000Z')
        self.assertFalse(path.exists())
        self.assertEqual(sorted(p.relative_to(archived).as_posix() for p in archived.rglob('*')), before)
        trainer = self.trainer(path, later)
        self.fixture.quiet_run(trainer)
        self.assertEqual(len([r for r in self.fixture.rows(path) if r['phase'] == 'train']), 2)

    def test_checkpointed_run_resumes_from_last_checkpoint(self):
        path = self.fixture.root / 'run'
        identity = self.identity()
        whole = self.trainer(self.fixture.root / 'whole', identity); self.fixture.quiet_run(whole)
        part = self.trainer(path, identity); self.fixture.quiet_run(part, stop_after=1)
        resume, archived = prepare_auto_resume(path, identity)
        self.assertEqual((resume, archived), (path / 'last.ckpt', None))
        self.fixture.quiet_run(self.trainer(path, identity, resume=resume))
        a = load_checkpoint(whole.run_dir / 'last.ckpt', identity)
        b = load_checkpoint(path / 'last.ckpt', identity)
        for key in ('model', 'optimizer', 'rng', 'progress'):
            self.fixture.assert_nested_equal(a[key], b[key])

    def test_other_directories_are_refused_untouched(self):
        cases = {
            'other_experiment': lambda p: (p / 'run.json').write_text(json.dumps(dict(
                run_id='x', identity=self.identity(model=dict(self.cfg['model'], variant='unet'))))),
            'checkpoint_without_last': lambda p: ((p / 'run.json').write_text(json.dumps(dict(
                run_id='x', identity=self.identity()))), (p / 'best-dev.ckpt').write_bytes(b'x')),
            'no_run_json': lambda p: (p / 'notes.txt').write_text('mine'),
            'broken_run_json': lambda p: (p / 'run.json').write_text('{'),
        }
        for name, make in cases.items():
            with self.subTest(case=name):
                path = self.fixture.root / name
                path.mkdir()
                make(path)
                before = {p.name: p.read_bytes() for p in path.iterdir()}
                with self.assertRaisesRegex(ValueError, 'not touched'):
                    prepare_auto_resume(path, self.identity(), now=NOW)
                self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)
                self.assertEqual(list(self.fixture.root.glob(f'{name}.aborted-*')), [])

    def test_cli_rejects_auto_resume_outside_training_or_with_resume(self):
        spec = importlib.util.spec_from_file_location('brats_auto_resume_test', fixtures.ROOT / 'scripts/brats.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        run = str(self.fixture.root / 'cli')
        for argv in (['check', '--auto-resume', '--shape', '5', '7', '9'],
                     ['train', '--auto-resume', '--run-dir', run, '--resume', run + '/last.ckpt'],
                     ['train', '--auto-resume']):
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(module.main(argv), 2)
                self.assertIn('auto-resume', stderr.getvalue())


if __name__ == '__main__':
    unittest.main()
