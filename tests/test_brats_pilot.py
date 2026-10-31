"""Metadata-only pilot selection, leakage guards and resumable dev monitoring."""
import contextlib
import copy
import importlib.util
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_training as fixtures
from test_brain_tumor import Cases
from organ_relation.data.brats import make_pilot_partition, validate_partition
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer
from organ_relation.training.state import capture_rng, digest, load_checkpoint, seed_all


def script(name):
    spec = importlib.util.spec_from_file_location(name, fixtures.ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class BraTSPilotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = json.loads((fixtures.ROOT/'configs/train_brats_patient_pilot.json').read_text())
        self.manifest = dict(cohort='brats2023_gli', source='read-only.zip', records=[
            dict(case_id=f'case-{p}-{t}', subject_id=f'patient-{p}', files={'seg':'unopened'})
            for p in range(8) for t in range(2)])
        self.kwargs = dict(train_subjects=2, dev_subjects=2, seed=20260925,
                          excluded_subjects=['patient-0'])
        self.split = make_pilot_partition(self.manifest, **self.kwargs)

    def test_selection_is_deterministic_nonmutating_and_patient_disjoint(self):
        before = copy.deepcopy(self.manifest);rng = capture_rng(torch.device('cpu'))
        self.assertEqual(self.split, make_pilot_partition(self.manifest, **self.kwargs))
        self.fixture.assert_nested_equal(rng, capture_rng(torch.device('cpu')))
        self.assertEqual(self.manifest, before)
        train, dev = validate_partition(self.manifest, self.split)
        self.assertEqual(len(train), 2);self.assertEqual(len(dev), 2)
        self.assertFalse({r['subject_id'] for r in train} & {r['subject_id'] for r in dev})
        self.assertNotIn('patient-0', [r['subject_id'] for r in train+dev])
        self.assertTrue(all(r['case_id'].endswith('-0') for r in train+dev))
        all_ids = self.split['train']+self.split['dev']+self.split['unused']
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {r['case_id'] for r in self.manifest['records']})
        reversed_manifest = dict(self.manifest, records=list(reversed(self.manifest['records'])))
        reordered = make_pilot_partition(reversed_manifest, **self.kwargs)
        for key in ('train','dev','unused'):
            self.assertEqual(self.split[key], reordered[key])

    def test_rejects_tampering_leakage_omissions_and_stale_source(self):
        changes = [dict(unused=[]), dict(train=self.split['dev']), dict(seed=42),
                   dict(train_subjects=3), dict(excluded_subjects=['missing']), dict(kind='unknown'),
                   dict(manifest_hash='stale'), dict(scan_policy='label_based')]
        # A second timepoint of a training patient cannot be put in dev.
        changes.append(dict(dev=[self.split['train'][0][:-1]+'1', self.split['dev'][1]]))
        for fields in changes:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_partition(self.manifest, dict(self.split, **fields))
        ordinary = dict(manifest_hash=digest(self.manifest), train=self.split['train'], dev=self.split['dev'])
        with self.assertRaises(ValueError):validate_partition(self.manifest, ordinary)
        with self.assertRaises(ValueError):validate_partition(self.manifest, dict(ordinary,unused=self.split['unused']))
        for fields in (dict(train_subjects=0),dict(dev_subjects=True),dict(seed=1.5),
                       dict(train_subjects=100),dict(excluded_subjects=['missing'])):
            with self.assertRaises(ValueError):make_pilot_partition(self.manifest, **dict(self.kwargs, **fields))
        with self.assertRaises(ValueError):
            make_pilot_partition(dict(self.manifest, cohort='brats2024_gli_post'), **self.kwargs)

    def test_config_and_training_plan_preserve_science_and_full_source_identity(self):
        original = json.loads((fixtures.ROOT/'configs/train_brats_single_case_overfit.json').read_text())
        for key in original.keys()-{'training','status'}:
            self.assertEqual(original[key], self.cfg[key])
        self.assertEqual(self.cfg['training'], dict(original['training'],
            validation_role='internal_dev', validation_every=80))
        path = self.fixture.root/'split.json';path.write_text(json.dumps(self.split))
        args = SimpleNamespace(action='train',case=None,split=path,run_dir=self.fixture.root/'run',max_steps=800)
        train, dev, options, data = script('brats').training_plan(args, self.manifest, self.cfg)
        self.assertEqual([r['case_id'] for r in train], self.split['train'])
        self.assertEqual([r['case_id'] for r in dev], self.split['dev'])
        self.assertEqual(data['manifest'], self.manifest)
        self.assertEqual(data['split_hash'], digest(self.split))
        self.assertEqual(options['max_steps'], 800)

    def test_split_cli_reads_metadata_and_refuses_overwrite(self):
        output = self.fixture.root/'pilot.local.json'
        args = ['--source','read-only.zip','--output',str(output),'--train-subjects','2',
                '--dev-subjects','2','--seed','20260925','--exclude-subject','patient-0']
        cli = script('brats_pilot_split')
        with patch.object(cli,'index_cases',return_value=self.manifest), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args), 0)
            content = output.read_bytes()
            self.assertEqual(json.loads(content), self.split)
            self.assertEqual(cli.main(args), 2)
            self.assertEqual(output.read_bytes(), content)

    def trainer(self, path, resume=False):
        seed_all(self.cfg['training']['seed'])
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']),
            convolution_backend='depth_sliced').to(memory_format=torch.channels_last_3d)
        options = dict(self.cfg['training'],max_steps=4,checkpoint_every=1,validation_every=2)
        identity = dict(mode='brats_regions',training=options,data=dict(manifest=self.manifest,
                        manifest_hash=digest(self.manifest),split=self.split,split_hash=digest(self.split)),
                        model=self.cfg['model'],loss=self.cfg['loss'],threshold=self.cfg['threshold'],
                        preprocessing=self.cfg['preprocessing'],provenance='CPU pilot fixture')
        optimizer = torch.optim.AdamW(model.parameters(), **self.cfg['optimizer'])
        return Trainer(model,RegionJointLoss(**self.cfg['loss']),optimizer,Cases(),self.split['train'],
            device='cpu',options=options,identity=identity,run_dir=path,validation_dataset=Cases(),
            validation_case_ids=self.split['dev'],resume=path/'last.ckpt' if resume else None)

    def test_multicase_resume_keeps_trajectory_and_dev_steps(self):
        whole = self.trainer(self.fixture.root/'whole');self.fixture.quiet_run(whole)
        part = self.trainer(self.fixture.root/'part');self.fixture.quiet_run(part,stop_after=1)
        resumed = self.trainer(part.run_dir,True);self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b = load_checkpoint(part.run_dir/'last.ckpt',resumed.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        rows = self.fixture.rows(part.run_dir)
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train'], [1,2,3,4])
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='internal_dev'], [2,4])
        self.assertEqual({r['case_id'] for r in rows if r['phase']=='train'},set(self.split['train']))
        for row in (r for r in rows if r['phase']=='internal_dev'):
            self.assertEqual(set(row['region_diagnostic_cases']),set(self.split['dev']))
        changed = copy.deepcopy(resumed.identity);changed['data']['split']['seed'] += 1
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            load_checkpoint(part.run_dir/'last.ckpt',changed)


if __name__ == '__main__':
    unittest.main()
