"""MONAI bottleneck adapter identity and proposed integration acceptance."""
import torch
import json
import copy
import inspect
from unittest.mock import patch
from monai.losses import DiceCELoss
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from test_monai_reference import Cases
import test_monai_reference as reference_tests
from organ_relation.training.monai_relation import MonaiRelationLoss
from organ_relation.training.engine import Trainer, gradient_diagnostics
from organ_relation.training.state import load_checkpoint
import unittest

from test_monai_reference import config
import test_training as fixtures
from organ_relation.models.monai_reference import MonaiReferenceUNet
from organ_relation.models.monai_relation import MonaiRelationUNet, BOTTLENECK_PATH
from organ_relation.training.state import seed_all


class MonaiRelationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = config()

    def test_bypass_output_input_and_all_backbone_gradients_exact(self):
        for fmt in (torch.contiguous_format, torch.channels_last_3d):
            seed_all(17)
            original = MonaiReferenceUNet(self.cfg['model'], self.cfg['input_processing']).to(memory_format=fmt)
            seed_all(17)
            wrapped = MonaiRelationUNet(self.cfg['model'], self.cfg['input_processing'], relation_enabled=False).to(memory_format=fmt)
            x = (torch.randn(1,1,17,19,21)*500).contiguous(memory_format=fmt).requires_grad_()
            y = x.detach().clone().requires_grad_()
            a, b = original(x).final_logits, wrapped(y).final_logits
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            weight = torch.randn_like(a)
            (a*weight).mean().backward(); (b*weight).mean().backward()
            torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            reference = dict(original.named_parameters())
            for name,p in wrapped.named_parameters():
                mapped=name.replace(BOTTLENECK_PATH+'.block.',BOTTLENECK_PATH+'.')
                torch.testing.assert_close(p,reference[mapped],rtol=0,atol=0)
                torch.testing.assert_close(p.grad,reference[mapped].grad,rtol=0,atol=0)
            self.assertEqual(len(reference),len(list(wrapped.parameters())))
            self.assertTrue(all(not m._forward_hooks for m in wrapped.modules()))


    def proposed_config(self):
        return json.loads((fixtures.ROOT/'configs/train_monai_relation_whole_volume_overfit.json').read_text())

    def model(self):
        cfg = self.proposed_config()
        return MonaiRelationUNet(cfg['model'],cfg['input_processing'],relation_enabled=True,relation_config=cfg['relation'])

    def criterion(self):
        cfg=self.proposed_config()
        return MonaiRelationLoss(cfg['loss'],**cfg['coarse_supervision'])

    def test_config_only_intended_additions_and_seeded_backbone_unchanged(self):
        cfg=self.proposed_config()
        for key,value in self.cfg.items():
            if key not in ('mode','purpose'): self.assertEqual(cfg[key],value,key)
        seed_all(9); original=MonaiReferenceUNet(self.cfg['model'],self.cfg['input_processing'])
        seed_all(9); wrapped=self.model()
        original_params=dict(original.named_parameters())
        seen=[]
        for n,p in wrapped.named_parameters():
            mapped=n.replace(BOTTLENECK_PATH+'.block.',BOTTLENECK_PATH+'.')
            if mapped in original_params:
                seen.append(mapped);torch.testing.assert_close(p,original_params[mapped],rtol=0,atol=0)
        self.assertEqual(set(seen),set(original_params))
        with self.assertRaisesRegex(ValueError,'feature_channels'):
            MonaiRelationUNet(cfg['model'],cfg['input_processing'],relation_enabled=True,
                              relation_config=dict(cfg['relation'],feature_channels=32))

    def test_enabled_shapes_nodes_edges_gradients_and_no_label_input(self):
        m=self.model().to(memory_format=torch.channels_last_3d); b=m.bottleneck
        image=torch.randn(1,1,17,19,21)*500
        nodes=[];h=b.space_to_node.register_forward_hook(lambda m,i,o:nodes.append(o))
        try: out=m(image)
        finally:h.remove()
        self.assertEqual(out.final_logits.shape,(1,16,17,19,21))
        self.assertEqual(out.coarse_logits.shape,(1,16,2,2,2))
        node=nodes[0];self.assertEqual(node.z0.shape,(1,15,128))
        relation=b.relation(node.z0,node.centroid,node.size,node.confidence,return_diagnostics=True)
        for step in relation.rounds:
            self.assertEqual(int((step.alpha!=0).sum()),210)
            self.assertTrue((step.alpha.diagonal(dim1=1,dim2=2)==0).all())
        label=torch.arange(17*19*21).reshape(1,17,19,21)%16
        self.criterion()(out.coarse_logits,out.final_logits,label).total.backward()
        groups=gradient_diagnostics(m)
        self.assertEqual(set(groups),{'encoder','decoder','coarse_head','relation','node_to_space','fusion'})
        self.assertTrue(all(v['finite'] and v['gradient_norm']>0 for v in groups.values()))
        grouped=[id(p) for ps in m.diagnostic_parameter_groups().values() for _,p in ps]
        self.assertEqual(len(grouped),len(set(grouped)))
        self.assertEqual(set(grouped),{id(p) for p in m.parameters()})
        self.assertEqual(list(inspect.signature(m.forward).parameters),['image'])
        with self.assertRaises(TypeError):m(image,label=label)
        self.assertIsNone(b._coarse_sink)
        self.assertTrue(all(not mod._forward_hooks for mod in m.modules()))

    def test_scoped_collector_cleared_on_failure_and_next_forward(self):
        m=self.model();x=torch.randn(1,1,17,19,21)
        with patch.object(m.bottleneck.fusion,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):m(x)
        self.assertIsNone(m.bottleneck._coarse_sink)
        self.assertEqual(m(x).final_logits.shape,(1,16,17,19,21))
        self.assertIsNone(m.bottleneck._coarse_sink)

    def test_official_joint_loss_logits_first_and_both_gradients(self):
        criterion=self.criterion();official=DiceCELoss(**self.cfg['loss'])
        c=torch.randn(1,16,2,3,2,requires_grad=True)
        f=torch.randn(1,16,5,7,6,requires_grad=True)
        label=torch.arange(210).reshape(1,5,7,6)%16;original=label.clone()
        with patch.object(criterion.branch.loss,'forward',wraps=criterion.branch.loss.forward) as call:
            actual=criterion(c,f,label)
        self.assertEqual(call.call_count,2)
        aligned=F.interpolate(c,size=(5,7,6),mode='trilinear',align_corners=False)
        torch.testing.assert_close(call.call_args_list[1].args[0],aligned,rtol=0,atol=0)
        expected=official(f,label.unsqueeze(1))+.5*official(aligned,label.unsqueeze(1))
        torch.testing.assert_close(actual.total,expected,rtol=0,atol=0)
        for a,b in zip(torch.autograd.grad(actual.total,(c,f),retain_graph=True),torch.autograd.grad(expected,(c,f))):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(label,original))
        self.assertEqual(actual.coarse.dice_per_class.shape,(1,15))

    def trainer(self,path,resume=False):
        cfg=self.proposed_config();seed_all(171)
        m=self.model();loss=self.criterion()
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0,foreach=False,fused=False)
        options=copy.deepcopy(cfg['training']);options.update(max_steps=2,validation_every=1,diagnostics_every=1,checkpoint_every=1)
        ident=dict(mode=cfg['mode'],training=options,model=cfg['model'],loss=cfg['loss'],
                   relation=cfg['relation'],relation_enabled=True,coarse_supervision=cfg['coarse_supervision'],
                   preprocessing=cfg['input_processing'],provenance='test',data={'manifest_hash':'synthetic'})
        return Trainer(m,loss,opt,Cases(),['one','two'],device='cpu',options=options,identity=ident,
            run_dir=path,validation_dataset=Cases(),validation_case_ids=['one','two'],tensorboard=True,
            resume=path/'last.ckpt' if resume else None)

    def test_resume_exact_relation_optimizer_rng_case_order_metrics_tensorboard(self):
        whole=self.trainer(self.fixture.root/'whole');self.fixture.quiet_run(whole)
        first=self.trainer(self.fixture.root/'resumed');self.fixture.quiet_run(first,stop_after=1)
        second=self.trainer(first.run_dir,True);self.fixture.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        self.assertTrue(any('.relation.' in k for k in b['model']))
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        for x,y in zip(self.fixture.rows(whole.run_dir),self.fixture.rows(second.run_dir)):
            for key in ('phase','global_step','case_id','total_loss','final','coarse','metrics','diagnostic_cases','geometry'):
                self.assertEqual(x.get(key),y.get(key))
            if x['phase']=='train': self.assertIn('update_norm',x['diagnostics']['relation'])
        self.assertFalse(second.board.failed)
        ev=EventAccumulator(str(second.board.directory),size_guidance={'scalars':0}).Reload()
        for tag in ('Train/Total_Loss','Train/Coarse_CE','GradNorm/Relation',
                    'MonitorCoarse/total_loss','MonitorCoarse/HardDice_Mean','MonitorCoarse/final_soft_dice',
                    'Monitor/foreground_true_positive_voxels','MonitorDice/Liver','MonitorSoftDice/Liver'):
            self.assertEqual([e.step for e in ev.Scalars(tag)],[1,2],tag)
        changed=copy.deepcopy(second.identity);changed['relation']['rounds']=3
        with self.assertRaisesRegex(ValueError,'identity mismatch'):load_checkpoint(second.run_dir/'last.ckpt',changed)

    def test_cli_preflight_and_resume_with_relation_identity(self):
        # Reuse the real synthetic-NIfTI CLI acceptance, with this independent config.
        helper=reference_tests.MonaiReferenceTests();helper.setUp()
        try:
            helper.cfg=self.proposed_config()
            helper.test_cli_synthetic_nifti_preflight_then_resumable_run()
        finally:helper.doCleanups()

if __name__ == '__main__': unittest.main()
