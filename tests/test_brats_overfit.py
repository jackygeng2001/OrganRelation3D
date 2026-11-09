"""Single-case CLI selection and 5 -> 100 transactional CPU continuation."""
import copy
import contextlib
import importlib.util
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import support as fixtures
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
        self.cfg = json.loads((fixtures.ROOT/'configs/development/single_case_overfit.json').read_text())
        self.manifest = dict(records=[dict(case_id='one',subject_id='patient1'),
                                     dict(case_id='two',subject_id='patient1'),
                                     dict(case_id='three',subject_id='patient2')])
        self.args = SimpleNamespace(action='overfit',case='one',split=None,
                                    run_dir=self.fixture.root/'run',max_steps=100)

    def test_config_keeps_scientific_settings_and_explicit_monitor_schedule(self):
        original = json.loads((fixtures.ROOT/'configs/development/implementation_check.json').read_text())
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
        cfg = json.loads((fixtures.ROOT/'configs/development/implementation_check.json').read_text())
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

    def trainer(self,path,resume=False,*,extend_to=None,planned=100):
        seed_all(self.cfg['training']['seed'])
        model = BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']),
                                   convolution_backend='depth_sliced').to(memory_format=torch.channels_last_3d)
        _,_,options,data = self.cli.training_plan(self.args,self.manifest,self.cfg)
        options['max_steps']=planned
        identity = dict(mode='brats_regions',model=self.cfg['model'],loss=self.cfg['loss'],
                        threshold=self.cfg['threshold'],training=options,data=data,provenance='CPU fixture',
                        preprocessing=self.cfg['preprocessing'],
                        runtime=dict(convolution_backend='depth_sliced',convolution_depth_chunk=4))
        optimizer = torch.optim.AdamW(model.parameters(),**self.cfg['optimizer'])
        return Trainer(model,RegionJointLoss(**self.cfg['loss']),optimizer,SingleCase(),['one'],
                       device='cpu',options=options,identity=identity,run_dir=path,
                       validation_dataset=SingleCase(),validation_case_ids=['one'],tensorboard=True,
                       resume=path/'last.ckpt' if resume else None,extend_to=extend_to)

    def test_extension_cli_requires_resume_and_training_action(self):
        for args in (['overfit','--extend-to','500'],['check','--resume','missing','--extend-to','500'],
                     ['inspect','--resume','missing','--extend-to','500'],['overfit','--resume','missing','--extend-to','0']):
            with contextlib.redirect_stderr(io.StringIO()),patch('organ_relation.data.brats.index_cases',side_effect=AssertionError('too late')):
                self.assertEqual(self.cli.main(args),2)

    def test_brats_extension_to_500_preserves_updates_logs_and_tensorboard(self):
        # Actual BraTS model at small CPU geometry; bound this integration test
        # to five updates. The shared-engine test below executes the full 500.
        whole=self.trainer(self.fixture.root/'whole500',planned=500)
        self.fixture.quiet_run(whole,stop_after=5)
        part=self.trainer(self.fixture.root/'part500');self.fixture.quiet_run(part,stop_after=2)
        original_run=(part.run_dir/'run.json').read_bytes()
        extended=self.trainer(part.run_dir,True,extend_to=500)
        self.assertEqual(extended.total,500);self.assertEqual(extended.state['global_step'],2)
        self.fixture.quiet_run(extended,stop_after=2)
        normal=self.trainer(part.run_dir,True);self.fixture.quiet_run(normal,stop_after=1)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(normal.run_dir/'last.ckpt',normal.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        self.assertEqual((part.run_dir/'run.json').read_bytes(),original_run)
        self.assertEqual(normal.run_id,part.run_id)
        self.assertEqual(b['horizon']['original_total_steps'],100)
        self.assertEqual(b['horizon']['total_steps'],500)
        events=EventAccumulator(str(normal.board.directory),size_guidance={'scalars':0}).Reload()
        self.assertEqual([e.step for e in events.Scalars('Train/Total_Loss')],[1,2,3,4,5])

    def test_shared_engine_100_to_500_exact_full_trajectory(self):
        options=dict(max_steps=500,checkpoint_every=25,validation_every=25)
        whole=self.fixture.trainer(self.fixture.root/'full500',options=options,validation=True)
        self.fixture.quiet_run(whole)
        part=self.fixture.trainer(self.fixture.root/'extend500',options=dict(options,max_steps=100),validation=True)
        self.fixture.quiet_run(part)
        parent=load_checkpoint(part.run_dir/'last.ckpt',part.identity)
        self.assertEqual(parent['progress']['global_step'],100)
        extended=self.fixture.trainer(part.run_dir,resume=True,options=dict(options,max_steps=100),validation=True,extend_to=500)
        self.fixture.quiet_run(extended)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(part.run_dir/'last.ckpt',extended.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        rows=self.fixture.rows(part.run_dir)
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train'],list(range(1,501)))
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train_monitor'],list(range(25,501,25)))
        self.assertEqual([r['case_id'] for r in rows if r['phase']=='train'],
                         [r['case_id'] for r in self.fixture.rows(whole.run_dir) if r['phase']=='train'])
        event=b['horizon']['extensions'][0]
        self.assertEqual((event['at_global_step'],event['previous_total_steps'],event['total_steps']),(100,100,500))
        self.assertEqual(event['max_epochs'],parent['identity']['training']['max_epochs'])

    def test_brats_source_upgrade_is_fixed_and_science_stays_strict(self):
        from organ_relation.training.extension import check_extension_identity,BRATS_COMMIT
        old=dict(mode='brats_regions',model='fixed',optimizer=dict(lr=3e-4),
            training=dict(max_steps=100,max_epochs=None,seed=1),data=dict(selection=dict(kind='single_case_overfit'),split='fixed'),
            runtime=dict(convolution_backend='depth_sliced'),loss='fixed',preprocessing='fixed',environment='fixed',
            provenance=dict(git=dict(commit=BRATS_COMMIT,dirty=False),source_hashes={
                'scripts/brats.py':'old','scripts/run_tests.py':'old','src/organ_relation/training/extension.py':'old',
                'src/organ_relation/models/brain_tumor.py':'fixed','src/organ_relation/training/engine.py':'fixed'}))
        new=copy.deepcopy(old);new['provenance']['git']['commit']='extension-release'
        new['provenance']['source_hashes'].update({'scripts/brats.py':'new','scripts/brats_diagnose.py':'new',
            'scripts/run_tests.py':'new','src/organ_relation/training/extension.py':'new'})
        with patch('organ_relation.training.extension.BRATS_SOURCES',digest(old['provenance']['source_hashes'])):
            check_extension_identity(old,new)
            for key in ('model','loss','optimizer','data','training','runtime','environment','preprocessing'):
                bad=copy.deepcopy(new);bad[key]='changed'
                with self.assertRaisesRegex(ValueError,'identity mismatch'):check_extension_identity(old,bad)
            for path in ('src/organ_relation/models/brain_tumor.py','src/organ_relation/training/engine.py','src/organ_relation/losses.py','scripts/unreviewed.py'):
                bad=copy.deepcopy(new);bad['provenance']['source_hashes'][path]='changed'
                with self.assertRaisesRegex(ValueError,'source mismatch'):check_extension_identity(old,bad)
            for version in (dict(commit=BRATS_COMMIT,dirty=True),dict(commit='unknown',dirty=False)):
                bad=copy.deepcopy(old);bad['provenance']['git']=version
                with self.assertRaisesRegex(ValueError,'provenance mismatch'):check_extension_identity(bad,new)
            bad=copy.deepcopy(new);bad['provenance']['git']['dirty']=True
            with self.assertRaisesRegex(ValueError,'dirty checkout'):check_extension_identity(old,bad)
        bad=copy.deepcopy(old);bad['provenance']['source_hashes']['scripts/brats.py']='different'
        with self.assertRaisesRegex(ValueError,'provenance mismatch'):check_extension_identity(bad,new)

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
