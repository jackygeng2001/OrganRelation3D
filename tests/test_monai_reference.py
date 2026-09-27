"""Optional MONAI reference acceptance. No GPU, CT training or skipped tests."""
import contextlib
import copy
import importlib.util
import io
import json
import random
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import torch
from monai.losses import DiceCELoss, DiceLoss
from monai.networks.nets import UNet
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from organ_relation.data.full_scan import FullScanSample
from organ_relation.metrics import hard_dice
from organ_relation.models.monai_reference import MonaiReferenceUNet, padding_geometry
from organ_relation.training.monai_reference import MonaiReferenceLoss, reference_diagnostics, preflight_backward
from organ_relation.training.engine import Trainer
from organ_relation.training.state import load_checkpoint, seed_all, atomic_json

ROOT = fixtures.ROOT

def config():
    return json.loads((ROOT / 'configs/train_monai_reference_whole_volume_overfit.json').read_text())

class Cases(torch.utils.data.Dataset):
    def __len__(self): return 2
    def __getitem__(self, i):
        shape = (17 + i, 19, 21)
        x = torch.randn(1, *shape) * 500 + random.random() + float(np.random.random())
        y = torch.arange(np.prod(shape)).reshape(shape).long() % 16
        return FullScanSample(x, y, {})

class MonaiReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.cfg = config()

    def model(self):
        return MonaiReferenceUNet(self.cfg['model'], self.cfg['input_processing'])

    def test_exact_official_constructor_and_parameter_count(self):
        m = self.model()
        self.assertIsInstance(m.network, UNet)
        self.assertEqual(self.cfg['model']['channels'], [8, 16, 32, 64, 128])
        self.assertEqual(self.cfg['model']['strides'], [2, 2, 2, 2])
        seed_all(13); a = self.model()
        seed_all(13); b = UNet(**self.cfg['model'])
        self.fixture.assert_nested_equal(a.network.state_dict(), b.state_dict())
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        self.assertEqual(set(dict(a.named_children())), {'network'})

    def test_hu_and_minimal_padding_inverse_crop_whole_network_once(self):
        m = self.model().to(memory_format=torch.channels_last_3d)
        self.assertEqual(padding_geometry((150, 239, 239)), dict(original_shape=[150,239,239],
            padding_dhw=[[0,10],[0,1],[0,1]], padded_shape=[160,240,240]))
        for shape in ((17,19,21), (32,32,32)):
            x = torch.linspace(-1500,1500,np.prod(shape)).reshape(1,1,*shape)
            original = x.clone()
            x = x.contiguous(memory_format=torch.channels_last_3d)
            padded = m.prepare_image(x)
            torch.testing.assert_close(padded[:,:,:shape[0],:shape[1],:shape[2]], x.clamp(-1000,1000)/1000)
            torch.testing.assert_close(x, original)
            self.assertTrue(padded.is_contiguous(memory_format=torch.channels_last_3d))
            if shape[0] == 17: self.assertTrue((padded[:,:,17:] == -1).all())
            with patch.object(m.network, 'forward', wraps=m.network.forward) as call:
                output = m(x).final_logits
                self.assertEqual(call.call_count, 1)
                self.assertEqual(tuple(call.call_args.args[0].shape), (1,1,32,32,32))
            direct = m.network(padded)[:,:,:shape[0],:shape[1],:shape[2]]
            torch.testing.assert_close(output,direct,rtol=0,atol=0)
            self.assertEqual(output.shape,(1,16,*shape))
        with self.assertRaises(TypeError): m(x,label=torch.zeros(1))

    def test_official_loss_and_gradient_exact_no_padded_gt_or_custom_ce(self):
        adapter = MonaiReferenceLoss(self.cfg['loss'])
        direct = DiceCELoss(**self.cfg['loss'])
        logits = torch.randn(1,16,3,4,5,requires_grad=True)
        label = torch.arange(60).reshape(1,3,4,5).long()%16
        saved = label.clone()
        a = adapter(logits,label); b = direct(logits,label.unsqueeze(1))
        torch.testing.assert_close(a.total,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a.total,logits,retain_graph=True)[0]
        gb=torch.autograd.grad(b,logits)[0]
        torch.testing.assert_close(ga,gb,rtol=0,atol=0)
        torch.testing.assert_close(a.final.ce,direct.ce(logits,label.unsqueeze(1)).reshape(1))
        torch.testing.assert_close(a.final.dice_loss,direct.dice(logits,label.unsqueeze(1)).reshape(1))
        self.assertEqual(a.final.dice_per_class.shape,(1,15))
        self.assertTrue(torch.equal(label,saved))
        self.assertTrue(all(not m._forward_hooks for m in adapter.modules()))
        for key in ('ce_background_weight','ce_foreground_weight','foreground_ce_reduction'):
            self.assertFalse(hasattr(adapter,key))

    def test_complete_small_network_all_gradients_and_monitor(self):
        seed_all(20260925); m=self.model(); criterion=MonaiReferenceLoss(self.cfg['loss'])
        sample=Cases()[0]; image=sample.image.unsqueeze(0); label=sample.label.unsqueeze(0)
        out=m(image).final_logits; result=criterion(out,label); result.total.backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        with torch.no_grad():
            hard=hard_dice(out.argmax(1)[0],label[0])
            stats=reference_diagnostics(out,label,criterion,hard)
        self.assertEqual(len(stats['soft_dice_per_organ']),15)
        self.assertEqual(stats['foreground_true_positive_voxels'],sum(o['true_positive'] for o in hard['organs']))
        self.assertEqual(stats['predicted_foreground_voxels'],int((out.argmax(1)>0).sum()))
        self.assertAlmostEqual(stats['total_loss'],stats['ce_loss']+stats['dice_loss'],places=5)

    def trainer(self, path, resume=False):
        seed_all(171)
        m=self.model(); loss=MonaiReferenceLoss(self.cfg['loss'])
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0,foreach=False,fused=False)
        options=copy.deepcopy(self.cfg['training']);options.update(max_steps=2,validation_every=1,diagnostics_every=1,checkpoint_every=1)
        ident=dict(mode='monai_reference_unet',training=options,model=self.cfg['model'],loss=self.cfg['loss'],
                   preprocessing=self.cfg['input_processing'],provenance='test',data={'manifest_hash':'synthetic'})
        return Trainer(m,loss,opt,Cases(),['one','two'],device='cpu',options=options,identity=ident,
            run_dir=path,validation_dataset=Cases(),validation_case_ids=['one','two'],tensorboard=True,
            resume=path/'last.ckpt' if resume else None)

    def test_resume_exact_optimizer_rng_metrics_and_tensorboard(self):
        whole=self.trainer(self.root/'whole');self.fixture.quiet_run(whole)
        first=self.trainer(self.root/'resumed');self.fixture.quiet_run(first,stop_after=1)
        second=self.trainer(first.run_dir,True);self.fixture.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        for x,y in zip(self.fixture.rows(whole.run_dir),self.fixture.rows(second.run_dir)):
            for key in ('phase','global_step','case_id','total_loss','final','metrics','diagnostic_cases','geometry'):
                self.assertEqual(x.get(key),y.get(key))
            self.assertNotIn('coarse',y)
        ev=EventAccumulator(str(second.board.directory),size_guidance={'scalars':0}).Reload()
        self.assertFalse(second.board.failed)
        for tag in ('Train/Total_Loss','Monitor/total_loss','Monitor/foreground_true_positive_voxels',
                    'Monitor/gt_foreground_true_class_mean_probability','MonitorSoftDice/Liver','MonitorDice/Liver'):
            self.assertEqual([e.step for e in ev.Scalars(tag)],[1,2],tag)
        changed=copy.deepcopy(second.identity);changed['loss']['smooth_nr']=1e-4
        with self.assertRaisesRegex(ValueError,'identity mismatch'):load_checkpoint(second.run_dir/'last.ckpt',changed)

    def test_cli_synthetic_nifti_preflight_then_resumable_run(self):
        spec=importlib.util.spec_from_file_location('monai_cli',ROOT/'scripts/train.py')
        cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        data=self.root/'data';data.mkdir();fixtures.synthetic_pair(data,shape=(48,48,48))
        for folder in ('imagesTr','labelsTr'):
            shutil.copyfile(data/folder/'amos_0001.nii.gz',data/folder/'amos_0002.nii.gz')
        atomic_json(data/'dataset.json',{'training':[dict(image=f'imagesTr/amos_{i:04}.nii.gz',label=f'labelsTr/amos_{i:04}.nii.gz') for i in (1,2)]})
        selection=json.loads((ROOT/'configs/ct_stats.json').read_text())
        artifact=fixtures.create_development(fixtures.training_manifest(data,selection),17,1)
        split=self.root/'split.json';fixtures.write_split(split,artifact)
        cfg=self.cfg
        for key in ('baseline_config','selection_config'):cfg[key]=str(ROOT/'configs'/cfg[key])
        cfg['runtime'].update(backend='cpu',device='cpu',cpu_threads=1)
        cfg['data'].update(train_cases=None,train_limit=1)
        cfg['training'].update(max_steps=2,validation_every=1,checkpoint_every=1)
        path=self.root/'config.json';atomic_json(path,cfg)
        argv=['--config',str(path),'--data-root',str(data),'--split',str(split),'--cpu-synthetic','--quiet-console']
        before={p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with contextlib.redirect_stdout(io.StringIO()) as out, patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('preflight must not step')):
            self.assertEqual(cli.main(argv+['--preflight-backward']),0)
        self.assertIn('"status": "passed"',out.getvalue())
        self.assertEqual(before,{p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        run=self.root/'run'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(argv+['--run-dir',str(run),'--stop-after','1']),0)
            self.assertEqual(cli.main(argv+['--run-dir',str(run),'--resume',str(run/'last.ckpt')]),0)
        identity=json.loads((run/'run.json').read_text())['identity']
        self.assertEqual(identity['mode'],'monai_reference_unet')
        self.assertEqual(identity['environment']['packages']['monai'],'1.6.0')
        self.assertIn('reference_after_hu',identity['preprocessing']);self.assertIn('geometry',identity)
        self.assertNotIn('ce_background_weight',identity)
        cfg['ce_reduction_mode']='foreground_background_balanced';atomic_json(path,cfg)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(argv+['--preflight-backward']),2)

    def test_preflight_oom_no_retry_or_parameter_update(self):
        m=self.model(); before={k:v.clone() for k,v in m.state_dict().items()}
        identity=dict(mode='monai_reference_unet',parameter_count=sum(p.numel() for p in m.parameters()),
            model=self.cfg['model'],loss=self.cfg['loss'],preprocessing={},environment={},
            provenance={'git':{}},training=self.cfg['training'])
        with contextlib.redirect_stdout(io.StringIO()),patch.object(m,'forward',side_effect=torch.OutOfMemoryError('synthetic OOM')) as call:
            with self.assertRaises(torch.OutOfMemoryError):
                preflight_backward(m,MonaiReferenceLoss(self.cfg['loss']),Cases(),identity)
        self.assertEqual(call.call_count,1)
        self.fixture.assert_nested_equal(before,m.state_dict())

if __name__=='__main__': unittest.main()
