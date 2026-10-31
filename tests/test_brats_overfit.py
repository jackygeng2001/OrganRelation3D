"""Single-case CLI selection and 5 -> 100 transactional CPU continuation."""
import copy
import importlib.util
import json
from types import SimpleNamespace
import unittest

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from test_brain_tumor import Cases
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all, digest, load_checkpoint


class SingleCase(Cases):
    def __len__(self):
        return 1


class BraTSOverfitTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        spec = importlib.util.spec_from_file_location('overfit_cli',fixtures.ROOT/'scripts/brats.py')
        self.cli = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.cli)
        self.cfg = json.loads((fixtures.ROOT/'configs/train_brats_single_case_overfit.json').read_text())
        self.manifest = dict(records=[dict(case_id='one',subject_id='patient1'),
                                     dict(case_id='two',subject_id='patient1'),
                                     dict(case_id='three',subject_id='patient2')])
        self.args = SimpleNamespace(action='overfit',case='one',split=None,
                                    run_dir=self.fixture.root/'run',max_steps=100)

    def test_config_keeps_scientific_settings_and_explicit_monitor_schedule(self):
        original = json.loads((fixtures.ROOT/'configs/brats_regions.json').read_text())
        for key in ('model','loss','preprocessing','optimizer','threshold','model_choice_note'):
            self.assertEqual(original[key],self.cfg[key])
        expected = dict(original['training'],diagnostics_every=5,validation_role='train_monitor',
                        console_every_steps=5)
        self.assertEqual(self.cfg['training'],expected)
        self.assertEqual(expected['checkpoint_every'],5)
        self.assertEqual(expected['validation_every'],25)

    def test_selection_is_one_case_and_explicitly_not_independent_validation(self):
        train,monitor,options,data = self.cli.training_plan(self.args,self.manifest,self.cfg)
        self.assertEqual(train,monitor)
        self.assertEqual([r['case_id'] for r in train],['one'])
        self.assertEqual(options['max_steps'],100)
        self.assertFalse(data['selection']['independent_validation'])
        self.assertNotIn('split',data)
        self.assertEqual(data['manifest_hash'],digest(self.manifest))

    def test_overfit_rejects_missing_case_split_ambiguous_case_and_wrong_role(self):
        for changes in (dict(case=None),dict(case='absent'),dict(split=self.fixture.root/'split.json'),
                        dict(max_steps=0),dict(run_dir=None)):
            args = copy.copy(self.args);vars(args).update(changes)
            with self.assertRaises(ValueError):
                self.cli.training_plan(args,self.manifest,self.cfg)
        repeated = dict(records=self.manifest['records']+[self.manifest['records'][0]])
        with self.assertRaises(ValueError):
            self.cli.training_plan(self.args,repeated,self.cfg)
        for changes in (dict(validation_role='internal_dev'),dict(max_epochs=5),dict(cadence_unit='epoch')):
            cfg = copy.deepcopy(self.cfg);cfg['training'].update(changes)
            with self.assertRaises(ValueError):
                self.cli.training_plan(self.args,self.manifest,cfg)

    def test_train_still_requires_exhaustive_patient_disjoint_split(self):
        cfg = json.loads((fixtures.ROOT/'configs/brats_regions.json').read_text())
        args = copy.copy(self.args);args.action='train'
        with self.assertRaises(ValueError):
            self.cli.training_plan(args,self.manifest,cfg)
        args.case=None;args.split=self.fixture.root/'split.json'
        for train,dev in ((['one'],['one']),(['one'],['two','three'])):
            args.split.write_text(json.dumps(dict(manifest_hash=digest(self.manifest),train=train,dev=dev)))
            with self.assertRaises(ValueError):
                self.cli.training_plan(args,self.manifest,cfg)
        args.split.write_text(json.dumps(dict(manifest_hash=digest(self.manifest),train=['one','two'],dev=['three'])))
        train,dev,_,data = self.cli.training_plan(args,self.manifest,cfg)
        self.assertEqual(len(train),2);self.assertEqual(len(dev),1)
        self.assertIn('split_hash',data);self.assertNotIn('selection',data)

    def trainer(self,path,resume=False):
        seed_all(self.cfg['training']['seed'])
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']),
                                   convolution_backend='depth_sliced').to(memory_format=torch.channels_last_3d)
        _,_,options,data = self.cli.training_plan(self.args,self.manifest,self.cfg)
        identity = dict(mode='brats_regions',model=self.cfg['model'],loss=self.cfg['loss'],
                        threshold=self.cfg['threshold'],training=options,data=data,provenance='CPU fixture',
                        preprocessing=self.cfg['preprocessing'],
                        runtime=dict(convolution_backend='depth_sliced',convolution_depth_chunk=4))
        optimizer = torch.optim.AdamW(model.parameters(),**self.cfg['optimizer'])
        return Trainer(model,RegionJointLoss(**self.cfg['loss']),optimizer,SingleCase(),['one'],
                       device='cpu',options=options,identity=identity,run_dir=path,
                       validation_dataset=SingleCase(),validation_case_ids=['one'],tensorboard=True,
                       resume=path/'last.ckpt' if resume else None)

    def test_five_step_stop_then_resume_to_100_preserves_state_logs_and_tensorboard(self):
        whole = self.trainer(self.fixture.root/'whole');self.fixture.quiet_run(whole)
        part = self.trainer(self.fixture.root/'part');self.fixture.quiet_run(part,stop_after=5)
        checkpoint = load_checkpoint(part.run_dir/'last.ckpt',part.identity)
        self.assertEqual(checkpoint['progress']['global_step'],5)
        self.assertEqual(part.total,100)
        step5 = self.fixture.rows(part.run_dir)[-1]
        self.assertTrue(all(v['finite'] for v in step5['diagnostics'].values()))
        self.assertTrue(all(v['update_norm']>0 for v in step5['diagnostics'].values()))
        resumed = self.trainer(part.run_dir,True);self.fixture.quiet_run(resumed)
        self.assertEqual(resumed.run_id,part.run_id)
        a = load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b = load_checkpoint(resumed.run_dir/'last.ckpt',resumed.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        rows = self.fixture.rows(resumed.run_dir)
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train'],list(range(1,101)))
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train_monitor'],[25,50,75,100])
        for x,y in zip(self.fixture.rows(whole.run_dir),rows):
            for key in ('global_step','case_id','total_loss','final','coarse','diagnostics','metrics','region_diagnostic_cases'):
                self.assertEqual(x.get(key),y.get(key))
        events = EventAccumulator(str(resumed.board.directory),size_guidance={'scalars':0}).Reload()
        self.assertEqual([e.step for e in events.Scalars('Monitor/HardDice/WT')],[25,50,75,100])
        for field,value in (('training',dict(resumed.identity['training'],max_steps=5)),
                            ('runtime',dict(convolution_backend='native')),
                            ('data',dict(resumed.identity['data'],selection={'train':['two']}))):
            changed = copy.deepcopy(resumed.identity);changed[field]=value
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                load_checkpoint(resumed.run_dir/'last.ckpt',changed)


if __name__ == '__main__':
    unittest.main()
