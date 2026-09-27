"""Identity and resumable training acceptance for the coarse-only ablation."""
import copy
import inspect
import json
import unittest
from unittest.mock import patch

import torch
from monai.losses import DiceCELoss
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.nn import functional as F

import test_training as fixtures
import test_monai_reference as reference_tests
from organ_relation.models.monai_coarse_aux import MonaiCoarseAuxUNet
from organ_relation.models.monai_reference import MonaiReferenceUNet
from organ_relation.models.monai_relation import BOTTLENECK_PATH
from organ_relation.training.engine import Trainer, gradient_diagnostics
from organ_relation.training.monai_relation import MonaiRelationLoss
from organ_relation.training.state import load_checkpoint, seed_all


def backbone_state(model):
    prefix = 'network.' + BOTTLENECK_PATH
    return {n.replace(prefix+'.block.', prefix+'.'): value
            for n, value in model.state_dict().items() if not n.startswith(prefix+'.coarse_head.')}


class MonaiCoarseAuxTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg = json.loads((fixtures.ROOT/'configs/train_monai_coarse_aux_whole_volume_overfit.json').read_text())

    def model(self):
        return MonaiCoarseAuxUNet(self.cfg['model'], self.cfg['input_processing'], coarse_head=self.cfg['coarse_head'])

    def criterion(self):
        return MonaiRelationLoss(self.cfg['loss'], **self.cfg['coarse_supervision'])

    def test_independent_config_init_state_and_parameter_count(self):
        reference = reference_tests.config()
        for key, value in reference.items():
            if key not in ('mode', 'purpose'):
                self.assertEqual(self.cfg[key], value, key)
        self.assertNotIn('relation_enabled', self.cfg)
        self.assertNotIn('relation', self.cfg)
        seed_all(self.cfg['training']['seed'])
        a = MonaiReferenceUNet(self.cfg['model'], self.cfg['input_processing'])
        seed_all(self.cfg['training']['seed']); b = self.model()
        self.fixture.assert_nested_equal(a.state_dict(), backbone_state(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 1215673)
        self.assertEqual(sum(p.numel() for p in b.bottleneck.coarse_head.parameters()), 2064)
        self.assertEqual(sum(p.numel() for p in b.parameters()), 1217737)
        groups = b.diagnostic_parameter_groups()
        self.assertEqual(set(groups), {'encoder', 'decoder', 'coarse_head'})
        ids = [id(p) for ps in groups.values() for _, p in ps]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in b.parameters()})

    def test_final_logits_input_backbone_gradients_exact_and_coarse_changes_encoder(self):
        for fmt in (torch.contiguous_format, torch.channels_last_3d):
            seed_all(17); a = MonaiReferenceUNet(self.cfg['model'], self.cfg['input_processing']).to(memory_format=fmt)
            seed_all(17); b = self.model().to(memory_format=fmt)
            x = (torch.randn(1,1,17,19,21)*500).contiguous(memory_format=fmt).requires_grad_()
            y = x.detach().clone().requires_grad_()
            target = torch.arange(17*19*21).reshape(1,17,19,21)%16
            before = {n:t.clone() for n,t in backbone_state(b).items()}
            ao, bo = a(x), b(y)
            torch.testing.assert_close(ao.final_logits, bo.final_logits, rtol=0, atol=0)
            criterion = self.criterion()
            criterion.branch(ao.final_logits,target).total.backward()
            criterion.branch(bo.final_logits,target).total.backward(retain_graph=True)
            torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            ap = dict(a.named_parameters()); prefix = 'network.'+BOTTLENECK_PATH
            for n,p in b.named_parameters():
                if n.startswith(prefix+'.coarse_head.'):
                    self.assertIsNone(p.grad)
                else:
                    mapped=n.replace(prefix+'.block.',prefix+'.')
                    torch.testing.assert_close(p.grad,ap[mapped].grad,rtol=0,atol=0)
            final_grads = {n:p.grad.clone() for n,p in b.named_parameters() if p.grad is not None}
            b.zero_grad(set_to_none=True)
            criterion(bo.coarse_logits,bo.final_logits,target).total.backward()
            groups=b.diagnostic_parameter_groups()
            self.assertTrue(any(not torch.equal(p.grad,final_grads[n]) for n,p in groups['encoder']))
            for n,p in groups['decoder']:
                torch.testing.assert_close(p.grad,final_grads[n],rtol=0,atol=0)
            self.assertTrue(all(v['finite'] and v['gradient_norm']>0 for v in gradient_diagnostics(b).values()))
            self.fixture.assert_nested_equal(before,backbone_state(b))  # No optimizer update.

    def test_decoder_receives_original_feature_object_no_relation_modules(self):
        m=self.model(); adapter=m.bottleneck; observed={}
        deepest_level=m.network.get_submodule(BOTTLENECK_PATH.rsplit('.1.submodule',1)[0])
        hooks=[adapter.block.register_forward_hook(lambda mod,args,out: observed.update(F=out)),
               adapter.register_forward_hook(lambda mod,args,out: observed.update(returned=out)),
               deepest_level[2].register_forward_pre_hook(lambda mod,args: observed.update(decoder=args[0]))]
        try: output=m(torch.randn(1,1,17,19,21))
        finally:
            for hook in hooks: hook.remove()
        self.assertIs(observed['F'],observed['returned'])
        torch.testing.assert_close(observed['decoder'][:,64:],observed['F'],rtol=0,atol=0)
        self.assertEqual(output.coarse_logits.shape,(1,16,2,2,2))
        self.assertEqual(output.final_logits.shape,(1,16,17,19,21))
        forbidden={'SpaceToNode','DynamicRelation','FormulaGRU','NodeToSpace','ResidualFusion'}
        self.assertFalse(forbidden & {type(mod).__name__ for mod in m.modules()})
        self.assertEqual(list(inspect.signature(m.forward).parameters),['image'])
        self.assertTrue(all(not mod._forward_hooks and not mod._forward_pre_hooks for mod in m.modules()))
        self.assertIsNone(adapter._coarse_sink)

    def test_official_joint_loss_upsamples_logits_to_unpadded_gt(self):
        loss=self.criterion(); direct=DiceCELoss(**self.cfg['loss'])
        coarse=torch.randn(1,16,2,3,2,requires_grad=True)
        final=torch.randn(1,16,5,7,6,requires_grad=True)
        label=torch.arange(210).reshape(1,5,7,6)%16; saved=label.clone()
        aligned=F.interpolate(coarse,size=(5,7,6),mode='trilinear',align_corners=False)
        with patch.object(loss.branch.loss,'forward',wraps=loss.branch.loss.forward) as call:
            actual=loss(coarse,final,label)
        self.assertEqual(call.call_count,2)
        torch.testing.assert_close(call.call_args_list[1].args[0],aligned,rtol=0,atol=0)
        expected=direct(final,label.unsqueeze(1))+.5*direct(aligned,label.unsqueeze(1))
        torch.testing.assert_close(actual.total,expected,rtol=0,atol=0)
        for a,b in zip(torch.autograd.grad(actual.total,(coarse,final),retain_graph=True),
                       torch.autograd.grad(expected,(coarse,final))):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(saved,label))

    def test_failed_forward_clears_scoped_state(self):
        m=self.model();x=torch.randn(1,1,17,19,21)
        with patch.object(m.bottleneck.coarse_head,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'): m(x)
        self.assertIsNone(m.bottleneck._coarse_sink)
        self.assertEqual(m(x).final_logits.shape,(1,16,17,19,21))
        self.assertIsNone(m.bottleneck._coarse_sink)

    def trainer(self,path,resume=False):
        seed_all(171); m=self.model(); loss=self.criterion()
        optimizer=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=0,foreach=False,fused=False)
        options=copy.deepcopy(self.cfg['training']); options.update(max_steps=2,validation_every=1,diagnostics_every=1,checkpoint_every=1)
        identity=dict(mode=self.cfg['mode'],training=options,model=self.cfg['model'],loss=self.cfg['loss'],
                      coarse_head=self.cfg['coarse_head'],coarse_supervision=self.cfg['coarse_supervision'],
                      preprocessing=self.cfg['input_processing'],provenance='test',data={'manifest_hash':'synthetic'})
        return Trainer(m,loss,optimizer,reference_tests.Cases(),['one','two'],device='cpu',options=options,
                       identity=identity,run_dir=path,validation_dataset=reference_tests.Cases(),
                       validation_case_ids=['one','two'],tensorboard=True,resume=path/'last.ckpt' if resume else None)

    def test_checkpoint_resume_exact_trajectory_and_monitoring_without_relation(self):
        whole=self.trainer(self.fixture.root/'whole'); self.fixture.quiet_run(whole)
        first=self.trainer(self.fixture.root/'resumed'); self.fixture.quiet_run(first,stop_after=1)
        second=self.trainer(first.run_dir,True); self.fixture.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        self.assertTrue(any('.coarse_head.' in key for key in b['model']))
        self.assertFalse(any('.relation.' in key for key in b['model']))
        left=self.fixture.rows(whole.run_dir); right=self.fixture.rows(second.run_dir)
        self.assertEqual(len(left),len(right))
        for x,y in zip(left,right):
            for key in ('phase','global_step','case_id','total_loss','final','coarse','metrics','diagnostic_cases','geometry'):
                self.assertEqual(x.get(key),y.get(key))
            if x['phase']=='train':
                self.assertEqual(set(x['diagnostics']),{'encoder','decoder','coarse_head'})
                self.assertIn('update_norm',x['diagnostics']['coarse_head'])
        self.assertFalse(second.board.failed)
        ev=EventAccumulator(str(second.board.directory),size_guidance={'scalars':0}).Reload()
        for tag in ('Train/Total_Loss','Train/Coarse_CE','GradNorm/CoarseHead','MonitorCoarse/total_loss',
                    'MonitorCoarse/HardDice_Mean','MonitorCoarse/final_soft_dice','MonitorCoarseDice/Liver',
                    'MonitorCoarseSoftDice/Liver','MonitorDice/Liver','MonitorSoftDice/Liver',
                    'Monitor/foreground_true_positive_voxels'):
            self.assertEqual([event.step for event in ev.Scalars(tag)],[1,2],tag)
        self.assertNotIn('GradNorm/Relation',ev.Tags()['scalars'])
        changed=copy.deepcopy(second.identity);changed['coarse_head']['bias']=False
        with self.assertRaisesRegex(ValueError,'identity mismatch'):load_checkpoint(second.run_dir/'last.ckpt',changed)
        changed=copy.deepcopy(second.identity);changed['mode']='monai_relation_unet'
        with self.assertRaisesRegex(ValueError,'identity mismatch'):load_checkpoint(second.run_dir/'last.ckpt',changed)

    def test_cli_synthetic_preflight_and_resume(self):
        helper=reference_tests.MonaiReferenceTests();helper.setUp()
        try:
            helper.cfg=self.cfg
            helper.test_cli_synthetic_nifti_preflight_then_resumable_run()
        finally: helper.doCleanups()


if __name__=='__main__': unittest.main()
