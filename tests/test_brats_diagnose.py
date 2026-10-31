"""Read-only checkpoint diagnostics: one forward, exact metrics, protected data."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image
import torch

from organ_relation.data.brats import index_cases
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.provenance import code_hashes, sha256
from organ_relation.training.state import digest, save_checkpoint

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('brats_diagnose',ROOT/'scripts/brats_diagnose.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_probability_statistics_match_direct_numpy_and_empty_regions(self):
        label=torch.tensor([[[[0,1,2,3]]]])
        p=torch.tensor([[[[[.1,.7,.6,.9]]],[[[.2,.8,.3,.9]]],[[[.4,.2,.1,.8]]]]])
        result=cli.probability_statistics(torch.logit(p),label,.5)
        wt=result['regions'][0]
        self.assertEqual((wt['predicted_voxels'],wt['gt_voxels'],wt['true_positive']),(3,3,3))
        self.assertAlmostEqual(wt['inside_gt']['mean'],np.mean([.7,.6,.9]),places=6)
        self.assertAlmostEqual(wt['outside_gt']['probability_sum'],.1,places=6)
        self.assertAlmostEqual(wt['outside_probability_mass_fraction'],.1/2.3,places=6)
        empty=cli.probability_statistics(torch.full((1,3,1,1,4),-10.),torch.zeros_like(label),.5)
        self.assertIsNone(empty['regions'][2]['inside_gt']['mean'])
        self.assertIsNone(empty['regions'][2]['dice'])
        json.dumps(empty,allow_nan=False)

    def fixture(self):
        cfg=json.loads((ROOT/'configs/brats_regions.json').read_text())
        source=self.root/'data';case='BraTS-GLI-00000-000';folder=source/case;folder.mkdir(parents=True)
        shape=(9,7,5)
        for role in ('t1n','t1c','t2w','t2f','seg'):
            array=(np.arange(np.prod(shape)).reshape(shape)%4 if role=='seg' else
                   np.arange(np.prod(shape)).reshape(shape)+1).astype(np.float32)
            (folder/(case+'-'+role+'.nii.gz')).write_bytes(gzip.compress(nib.Nifti1Image(array,np.eye(4)).to_bytes()))
        manifest=index_cases(source,cohort='brats2023_gli')
        identity=dict(mode='brats_regions',model=cfg['model'],loss=cfg['loss'],threshold=.5,
            preprocessing=cfg['preprocessing'],training=cfg['training'],
            data=dict(manifest=manifest,manifest_hash=digest(manifest)),
            runtime=dict(convolution_backend='depth_sliced',convolution_depth_chunk=4,dtype='float32',cpu_threads=2),
            provenance=dict(source_hashes=code_hashes(),git=dict(commit='cpu-fixture',dirty=False)))
        model=BrainTumorSegmentor(BrainTumorConfig(**cfg['model']),convolution_backend='depth_sliced')
        run=self.root/'run';run.mkdir()
        (run/'run.json').write_text(json.dumps(dict(identity=identity,run_id='test')))
        state=dict(schema_version=1,identity=identity,run_id='test',model=model.state_dict(),optimizer={},scheduler=None,
                   progress=dict(global_step=100),rng={},sampler_generator=None,loader_generator=None,log={})
        save_checkpoint(run/'last.ckpt',state)
        return run,source,case,identity

    def test_one_forward_read_only_real_nifti_checkpoint_and_overlay(self):
        run,source,case,_=self.fixture()
        tracked=[p for parent in (run,source) for p in parent.rglob('*') if p.is_file()]
        before={str(p):sha256(p) for p in tracked}
        calls=[];forward=BrainTumorSegmentor.forward
        def observe(model,image):
            self.assertFalse(model.training);self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(tuple(image.shape),(1,4,5,7,9));calls.append(1)
            return forward(model,image)
        out=io.StringIO();png=self.root/'overlay.png'
        with contextlib.redirect_stdout(out),contextlib.redirect_stderr(io.StringIO()),patch.object(
                BrainTumorSegmentor,'forward',observe),patch.object(torch.Tensor,'backward',side_effect=AssertionError('backward forbidden')),patch.object(
                torch.optim.AdamW,'step',side_effect=AssertionError('update forbidden')),patch.object(
                cli,'compare_monitor',side_effect=KeyError('legacy diagnostic field')):
            status=cli.main(['--checkpoint',str(run/'last.ckpt'),'--case',case,'--device','cpu','--overlay',str(png)])
        self.assertEqual(status,0);self.assertEqual(len(calls),1)
        self.assertEqual(before,{str(p):sha256(p) for p in tracked})
        report=json.loads(out.getvalue());self.assertEqual(report['global_step'],100)
        self.assertEqual(report['execution']['optimizer_steps'],0)
        self.assertEqual(report['monitor_comparison']['status'],'unavailable')
        self.assertIn('legacy diagnostic field',report['monitor_comparison']['reason'])
        with Image.open(png) as image:self.assertEqual(image.format,'PNG');self.assertGreater(image.width,1000)
        old=copy.deepcopy(report['loss_and_soft_dice']);old['irrelevant_diagnostic']='extra'
        row=dict(run_id='test',phase='train_monitor',global_step=100,region_diagnostic_cases={case:old},
                 metrics=dict(cases=1,mean_case_dice=report['final']['mean_hard_dice']))
        (run/'metrics.jsonl').write_text(json.dumps(row)+'\n')
        comparison=cli.compare_monitor(run,100,case,'test',report['loss_and_soft_dice'],report['final']['mean_hard_dice'])
        self.assertEqual(comparison['status'],'matched')
        old['total_loss']+=1;(run/'metrics.jsonl').write_text(json.dumps(row)+'\n')
        self.assertEqual(cli.compare_monitor(run,100,case,'test',report['loss_and_soft_dice'],report['final']['mean_hard_dice'])['status'],'mismatch')

    def test_source_manifest_and_checkpoint_integrity_rejected_before_forward(self):
        run,source,case,identity=self.fixture()
        changed=copy.deepcopy(identity);changed['provenance']['source_hashes']['src/organ_relation/losses.py']='wrong'
        with self.assertRaisesRegex(ValueError,'package source'):cli.verify_source(changed)
        file=next(source.rglob('*t1n.nii.gz'));file.write_bytes(file.read_bytes()+b'changed')
        with patch.object(BrainTumorSegmentor,'forward',side_effect=AssertionError('must reject before inference')),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['--checkpoint',str(run/'last.ckpt'),'--case',case,'--device','cpu']),2)
        checkpoint=run/'last.ckpt';checkpoint.write_bytes(checkpoint.read_bytes()[:-1])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['--checkpoint',str(checkpoint),'--case',case,'--device','cpu']),2)

    def test_overlay_cannot_overwrite_or_write_into_run_or_source(self):
        run=self.root/'run';source=self.root/'source';run.mkdir();source.mkdir()
        existing=self.root/'exists.png';existing.write_bytes(b'unchanged')
        for path in (run/'a.png',source/'a.png',existing,self.root/'a.ckpt'):
            with self.assertRaises(ValueError):cli.check_overlay_path(path,run,source)
        cli.check_overlay_path(self.root/'new.png',run,source)
        self.assertEqual(existing.read_bytes(),b'unchanged')


if __name__=='__main__':unittest.main()
