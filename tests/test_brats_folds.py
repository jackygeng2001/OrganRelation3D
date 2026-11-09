"""Legacy input, reproducible folds, region epoch monitoring and early-stop resume."""
import contextlib
import copy
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import nibabel as nib
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import support as fixtures
from test_brain_tumor import Cases
from test_brats_pilot import script
from organ_relation.data.brats import BraTSDataset, index_cases, make_folds, validate_partition
from organ_relation.data.full_scan import FullScanSample
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.training import brain_tumor as observations
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all, load_checkpoint, capture_rng, save_checkpoint


class FixedDevCases(Cases):
    """Like real unaugmented MRI, each dev case is independent of evaluation order."""
    def __getitem__(self, index):
        shape = (5+index, 7, 9)
        grid = torch.arange(np.prod(shape)).reshape(shape)
        return FullScanSample(torch.stack([(grid.float()+i)/100 for i in range(4)]),grid%4,{})


class BraTSFoldTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.cfg = json.loads((fixtures.ROOT/'configs/train_brats_year_fold.json').read_text())

    def source(self, extension='.nii', alias=False):
        folder=self.f.root/('raw'+extension); folder.mkdir()
        for split,case in [('TrainingData','BraTS20_Training_355'),('ValidationData','BraTS20_Validation_001')]:
            parent=folder/f'BraTS2020_{split}'/case;parent.mkdir(parents=True)
            for role in ('t1','t1ce','t2','flair','seg'):
                if role=='seg' and split=='ValidationData':continue
                grid=np.arange(60).reshape(3,4,5)
                a=(np.take([0,1,2,4],grid%4) if role=='seg' else grid).astype(np.float32)
                name='W39_1998.09.19_Segm.nii' if alias and role=='seg' else case+'_'+role+extension
                nib.save(nib.Nifti1Image(a,np.eye(4)),parent/name)
        return folder

    def test_raw_gzip_directory_zip_full_grid_equivalence(self):
        expected=None
        for ext,alias in [('.nii',True),('.nii.gz',False)]:
            folder=self.source(ext,alias)
            archive=folder.with_suffix('.zip')
            with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
                for p in folder.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(folder).as_posix())
            for source in (folder,archive):
                with self.assertRaisesRegex(ValueError,'incomplete'):index_cases(source,cohort='brats2020')
                m=index_cases(source,cohort='brats2020',partition='training')
                self.assertEqual(len(m['records']),1);self.assertEqual(m['source_partition'],'training')
                sample=BraTSDataset(m,**self.cfg['preprocessing'])[0]
                self.assertEqual(sample.image.shape,(4,5,4,3));self.assertEqual(set(sample.label.unique().tolist()),{0,1,2,3})
                if expected is not None:
                    torch.testing.assert_close(sample.image,expected.image,rtol=0,atol=0)
                    torch.testing.assert_close(sample.label,expected.label,rtol=0,atol=0)
                expected=sample
        folder=self.f.root/'raw.nii';p=next(folder.rglob('*Segm.nii'))
        with self.assertRaisesRegex(ValueError,'conflicts'):
            index_cases(folder,cohort='brats2018',partition='training')
        # A genuinely absent training label is an error, never a silent exclusion.
        raw=p.read_bytes();p.unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'):
            index_cases(folder,cohort='brats2020',partition='training')
        p.write_bytes(raw)
        p.with_name('BraTS20_Training_355_seg.nii').write_bytes(p.read_bytes())
        with self.assertRaisesRegex(ValueError,'duplicate'):index_cases(folder,cohort='brats2020',partition='training')
        with self.assertRaises(ValueError):index_cases(folder,cohort='brats2023_gli',partition='training')

    def test_patient_folds_deterministic_exhaustive_rng_neutral_and_audited(self):
        m=dict(cohort='brats2018',source='read-only',records=[dict(case_id=f'c{i:03d}',subject_id=f'p{i:03d}') for i in range(285)])
        rng=capture_rng(torch.device('cpu')); folds=make_folds(m,seed=20260925)
        self.f.assert_nested_equal(rng,capture_rng(torch.device('cpu')))
        self.assertEqual(folds,make_folds(m,seed=20260925))
        self.assertEqual(len({c for s in folds for c in s['dev']}),285)
        for s in folds:
            train,dev=validate_partition(m,s);self.assertEqual((len(train),len(dev)),(228,57))
            bad=copy.deepcopy(s);bad['train'][0],bad['dev'][0]=bad['dev'][0],bad['train'][0]
            with self.assertRaises(ValueError):validate_partition(m,bad)
        multi=dict(m,records=m['records']+[dict(case_id='timepoint2',subject_id='p000')])
        for s in make_folds(multi,seed=1):validate_partition(multi,s)
        cli=script('brats_folds');out=self.f.root/'folds'
        args=['--source','read-only','--cohort','brats2018','--output-dir',str(out),'--seed','20260925']
        with patch.object(cli,'index_cases',return_value=m),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(args),0);before=[p.read_bytes() for p in sorted(out.iterdir())]
            self.assertEqual(cli.main(args),2);self.assertEqual(before,[p.read_bytes() for p in sorted(out.iterdir())])

    def test_config_science_unchanged_and_epoch_horizon_is_explicit(self):
        pilot=json.loads((fixtures.ROOT/'configs/train_brats_patient_pilot.json').read_text())
        for key in ('model','loss','threshold','preprocessing','optimizer'):self.assertEqual(self.cfg[key],pilot[key])
        self.assertEqual(self.cfg['training']['max_epochs'],500)
        m=dict(cohort='brats2018',records=[dict(case_id=str(i),subject_id=str(i)) for i in range(5)])
        s=make_folds(m,seed=1)[0];path=self.f.root/'split.json';path.write_text(json.dumps(s))
        args=SimpleNamespace(action='train',case=None,split=path,run_dir=self.f.root/'run',max_steps=None)
        _,_,options,_=script('brats').training_plan(args,m,self.cfg)
        self.assertIsNone(options['max_steps']);self.assertEqual(options['max_epochs'],500)
        args.max_steps=100
        with self.assertRaisesRegex(ValueError,'forbids'):script('brats').training_plan(args,m,self.cfg)

    def trainer(self,path,resume=False):
        seed_all(42)
        model=BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']))
        options=dict(self.cfg['training'],max_epochs=5,memory_format='contiguous',checkpoint_every_steps=1,
                     early_stopping=dict(min_epochs=2,patience_epochs=2,min_delta=.001,monitor='dev_mean_region_hard_dice'))
        identity=dict(mode='brats_regions',training=options,model=self.cfg['model'],loss=self.cfg['loss'],
                      threshold=.5,preprocessing=self.cfg['preprocessing'],data=dict(manifest_hash='synthetic'),provenance='CPU fold test')
        return Trainer(model,RegionJointLoss(**self.cfg['loss']),torch.optim.AdamW(model.parameters(),**self.cfg['optimizer']),
                       Cases(),['a','b'],device='cpu',options=options,identity=identity,run_dir=path,
                       validation_dataset=FixedDevCases(),validation_case_ids=['v1','v2'],
                       tensorboard=True,resume=path/'last.ckpt' if resume else None)

    def test_region_epoch_early_stop_resume_identity_and_no_duplicate_observations(self):
        summarize=observations.summarize_regions
        def plateau(cases):return dict(summarize(cases),mean_case_dice=.5)
        with patch.object(observations,'summarize_regions',side_effect=plateau):
            whole=self.trainer(self.f.root/'whole');self.f.quiet_run(whole)
            part=self.trainer(self.f.root/'part');self.f.quiet_run(part,stop_after=3)
            resumed=self.trainer(part.run_dir,True);self.f.quiet_run(resumed)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(part.run_dir/'last.ckpt',resumed.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator','development_state','early_stopping'):
            self.f.assert_nested_equal(a[key],b[key])
        self.assertEqual(b['progress']['global_step'],6);self.assertTrue(b['early_stopping']['stopped'])
        best=load_checkpoint(part.run_dir/'best-dev.ckpt',resumed.identity)
        self.assertEqual(best['progress']['global_step'],2)
        # Crash during full-dev: replay only the unfinished case, then reach the
        # same stopped trajectory and monitor history as uninterrupted training.
        pending=self.trainer(self.f.root/'pending')
        original_batch=pending._batch
        def interrupted(dataset,index,*,validation=False):
            if validation and index==1:
                raise OSError('simulated validation interruption')
            return original_batch(dataset,index,validation=validation)
        with patch.object(observations,'summarize_regions',side_effect=plateau), \
                patch.object(pending,'_batch',side_effect=interrupted), self.assertRaises(OSError):
            self.f.quiet_run(pending)
        saved=load_checkpoint(pending.run_dir/'last.ckpt',pending.identity)
        self.assertTrue(saved['progress']['pending_validation'])
        recovered=self.trainer(pending.run_dir,True)
        with patch.object(observations,'summarize_regions',side_effect=plateau):
            self.f.quiet_run(recovered)
        recovered_state=load_checkpoint(pending.run_dir/'last.ckpt',recovered.identity)
        for key in ('model','optimizer','rng','progress','development_state','early_stopping'):
            self.f.assert_nested_equal(a[key],recovered_state[key])
        rows=self.f.rows(part.run_dir)
        self.assertEqual([r['global_step'] for r in rows if r['phase']=='train'],list(range(1,7)))
        for phase in ('internal_dev','train_epoch'):
            selected=[r for r in rows if r['phase']==phase];self.assertEqual([r['global_step'] for r in selected],[2,4,6])
            self.assertEqual(len(selected[-1]['epoch_metrics']['final']['soft_per_region']),3)
        events=EventAccumulator(str(part.run_dir/'tensorboard')).Reload()
        self.assertEqual([e.step for e in events.Scalars('Val/HardDice_Mean')],[2,4,6])
        self.assertEqual([e.step for e in events.Scalars('TrainEpoch/final/SoftDice/WT')],[2,4,6])
        changed=copy.deepcopy(resumed.identity);changed['training']['early_stopping']['patience_epochs']=3
        with self.assertRaises(ValueError):load_checkpoint(part.run_dir/'last.ckpt',changed)
        # A stopped run remains stopped; extension cannot bypass its termination policy.
        again=self.trainer(part.run_dir,True);self.f.quiet_run(again)
        self.assertEqual(again.state['global_step'],6)
        bad=copy.deepcopy(b);bad['early_stopping']['no_improvement_count']=0
        save_checkpoint(part.run_dir/'last.ckpt',bad)
        with self.assertRaisesRegex(ValueError,'validation history'):self.trainer(part.run_dir,True)


if __name__=='__main__':unittest.main()
