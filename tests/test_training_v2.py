"""Scalar-only MONAI objectives, epoch observers and resumable dev early stop."""
import contextlib
import copy
import io
import json
import statistics
import unittest
from unittest.mock import patch

import torch
from monai.losses import DiceCELoss
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
import test_development as development_tests
from test_monai_reference import Cases
from organ_relation.models.monai_relation import MonaiRelationUNet
from organ_relation.models.monai_reference import MonaiReferenceUNet
from organ_relation.training.monai_reference import MonaiReferenceLoss
from organ_relation.training.monai_relation import MonaiRelationLoss
from organ_relation.training.engine import Trainer
from organ_relation.training.console import TrainingConsole
from organ_relation.training.development import (advance_early_stopping, new_early_state,
                                                validate_early_stopping)
from organ_relation.training.state import seed_all, load_checkpoint, save_checkpoint

ROOT = fixtures.ROOT


def config(gated=False):
    name = 'relation_gated' if gated else 'reference'
    return json.loads((ROOT/f'configs/train_monai_{name}_A_160_40.json').read_text())


class TrainingV2Tests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.TrainingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.root = self.f.root

    def test_official_objectives_scalars_and_logits_gradients_bit_exact(self):
        cfg = config(True)
        final = torch.randn(1,16,3,4,5,requires_grad=True)
        coarse = torch.randn(1,16,2,2,3,requires_grad=True)
        label = torch.arange(60).reshape(1,3,4,5) % 16
        branch = MonaiReferenceLoss(cfg['loss'])
        joint = MonaiRelationLoss(cfg['loss'], **cfg['coarse_supervision'])
        old_a = branch(final,label)
        new_a = branch.objective(final,label)
        torch.testing.assert_close(old_a.total,new_a.total,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(old_a.total,final,retain_graph=True)[0],
                                   torch.autograd.grad(new_a.total,final,retain_graph=True)[0],rtol=0,atol=0)
        old = joint(coarse,final,label)
        with (patch.object(joint.branch.dice_per_class,'forward',side_effect=AssertionError('observer in objective')),
              patch.object(joint,'align_coarse',wraps=joint.align_coarse) as resize):
            new = joint.objective(coarse,final,label)
        self.assertEqual(resize.call_count,1)
        official = DiceCELoss(**cfg['loss'])
        expected_final = official(final,label.unsqueeze(1))
        expected_coarse = official(F.interpolate(coarse,size=label.shape[1:],mode='trilinear',align_corners=False),label.unsqueeze(1))
        for a,b in ((new.final,old.final.segmentation.squeeze()),(new.coarse,old.coarse.segmentation.squeeze()),
                    (new.total,old.total),(new.final,expected_final),(new.coarse,expected_coarse)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(new.total.ndim,0)
        for a,b in zip(torch.autograd.grad(old.total,(final,coarse),retain_graph=True),
                       torch.autograd.grad(new.total,(final,coarse))):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_real_gated_model_parameter_gradients_match_old_adapter(self):
        cfg=config(True); seed_all(cfg['training']['seed'])
        model=MonaiRelationUNet(cfg['model'],cfg['input_processing'],relation_enabled=True,relation_config=cfg['relation'])
        criterion=MonaiRelationLoss(cfg['loss'],**cfg['coarse_supervision'])
        sample=Cases()[0]; x=sample.image.unsqueeze(0); label=sample.label.unsqueeze(0)
        out=model(x);old=criterion(out.coarse_logits,out.final_logits,label)
        a=torch.autograd.grad(old.total,tuple(model.parameters()))
        out=model(x);new=criterion.objective(out.coarse_logits,out.final_logits,label)
        b=torch.autograd.grad(new.total,tuple(model.parameters()))
        torch.testing.assert_close(old.total,new.total,rtol=0,atol=0)
        for p,q in zip(a,b):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_real_baseline_parameter_gradients_match_old_adapter(self):
        cfg=config(); seed_all(cfg['training']['seed'])
        model=MonaiReferenceUNet(cfg['model'],cfg['input_processing'])
        criterion=MonaiReferenceLoss(cfg['loss'])
        sample=Cases()[0]; x=sample.image.unsqueeze(0); label=sample.label.unsqueeze(0)
        old=criterion(model(x).final_logits,label)
        a=torch.autograd.grad(old.total,tuple(model.parameters()))
        new=criterion.objective(model(x).final_logits,label)
        b=torch.autograd.grad(new.total,tuple(model.parameters()))
        torch.testing.assert_close(old.total,new.total,rtol=0,atol=0)
        for p,q in zip(a,b):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_only_one_ce_and_dice_objective_no_detached_duplicate(self):
        criterion=MonaiReferenceLoss(config()['loss'])
        x=torch.randn(1,16,2,3,4,requires_grad=True);label=torch.arange(24).reshape(1,2,3,4)%16
        with (patch.object(criterion.loss,'ce',wraps=criterion.loss.ce) as ce,
             patch.object(criterion.loss.dice,'forward',wraps=criterion.loss.dice.forward) as dice,
             patch.object(criterion.dice_per_class,'forward',side_effect=AssertionError('expensive observer'))):
            loss=criterion.objective(x,label);loss.total.backward()
        self.assertEqual(ce.call_count,1);self.assertEqual(dice.call_count,1)

    def test_early_stop_minimum_patience_and_inclusive_delta(self):
        options=config()['training'];validate_early_stopping(options)
        state=new_early_state()
        for epoch in range(5,101,5):
            state=advance_early_stopping(state,.5,epoch,options)
            self.assertEqual(state['stopped'],epoch==100)
        self.assertEqual(state['best_epoch'],5)
        self.assertEqual(state['no_improvement_count'],19)
        state=new_early_state()
        for epoch in range(5,101,5):state=advance_early_stopping(state,epoch/200,epoch,options)
        self.assertFalse(state['stopped'])
        for epoch in range(105,126,5):
            state=advance_early_stopping(state,.5,epoch,options)
            self.assertEqual(state['stopped'],epoch==125)
        state=new_early_state()
        state=advance_early_stopping(state,.5,5,options)
        state=advance_early_stopping(state,.5001,10,options)
        self.assertEqual(state['no_improvement_count'],0)
        self.assertEqual(state['patience_reference_metric'],.5001)
        self.assertEqual(state['best_epoch'],10)
        state=advance_early_stopping(state,.501,15,options)
        self.assertEqual(state['no_improvement_count'],0)
        with self.assertRaises(ValueError):advance_early_stopping(state,None,20,options)
        with self.assertRaises(ValueError):advance_early_stopping(state,.5,25,options)

    def test_cumulative_small_improvement_keeps_reference_until_threshold(self):
        options=config()['training']
        state=advance_early_stopping(new_early_state(),.5,5,options)
        for epoch,score,count in ((10,.50003,1),(15,.50006,2),(20,.50009,3)):
            state=advance_early_stopping(state,score,epoch,options)
            self.assertEqual(state['best_metric'],score)
            self.assertEqual(state['best_epoch'],epoch)
            self.assertEqual(state['patience_reference_metric'],.5)
            self.assertEqual(state['no_improvement_count'],count)
        state=advance_early_stopping(state,.5001,25,options)  # Exactly reference + min_delta.
        self.assertEqual(state['patience_reference_metric'],.5001)
        self.assertEqual(state['no_improvement_count'],0)
        state=advance_early_stopping(state,.50014,30,options)
        self.assertEqual(state['best_metric'],.50014)
        self.assertEqual(state['patience_reference_metric'],.5001)
        self.assertEqual(state['no_improvement_count'],1)
        state=advance_early_stopping(state,.49,35,options)
        self.assertEqual(state['best_metric'],.50014)
        self.assertEqual(state['best_epoch'],30)
        self.assertEqual(state['patience_reference_metric'],.5001)
        self.assertEqual(state['no_improvement_count'],2)

    def tiny_trainer(self,path,resume=False):
        seed_all(20260925); cfg=config(); opt=cfg['training']
        # Real min=100/patience=25 in a tiny three-case synthetic epoch.
        model=development_tests.TinyReference();criterion=MonaiReferenceLoss(cfg['loss'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0,foreach=False,fused=False)
        identity=dict(mode=cfg['mode'],training=opt,model='tiny',loss=cfg['loss'],optimizer=cfg['optimizer'],
            preprocessing='synthetic',provenance='test',data={'manifest_hash':'synthetic'})
        class Forty(fixtures.RandomCases):
            def __len__(self):return 40
            def __getitem__(self,index):return super().__getitem__(index%3)
        return Trainer(model,criterion,optimizer,fixtures.RandomCases(),['a','b','c'],device='cpu',options=opt,
            identity=identity,run_dir=path,validation_dataset=Forty(),validation_case_ids=[f'v{i}' for i in range(40)],
            resume=path/'last.ckpt' if resume else None,tensorboard=True)

    def test_early_stop_full_40_validation_resume_and_no_restart_after_stop(self):
        from organ_relation.metrics import summarize_dice
        def constant(records):
            self.assertEqual(len(records),40)
            summary=summarize_dice(records);summary['mean_case_dice']=.5
            return summary
        with patch('organ_relation.training.engine.summarize_dice',side_effect=constant):
            whole=self.tiny_trainer(self.root/'whole');self.f.quiet_run(whole)
            first=self.tiny_trainer(self.root/'resume');self.f.quiet_run(first,stop_after=285) # epoch95
            middle=load_checkpoint(first.run_dir/'last.ckpt',first.identity)
            self.assertFalse(middle['early_stopping']['stopped'])
            second=self.tiny_trainer(first.run_dir,True);self.f.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator','development_state','early_stopping'):
            self.f.assert_nested_equal(a[key],b[key])
        self.assertEqual(b['progress']['epoch'],100);self.assertEqual(b['progress']['global_step'],300)
        self.assertTrue(b['early_stopping']['stopped'])
        self.assertEqual(len(b['development_state']['validation_history']),20)
        stopped=self.tiny_trainer(first.run_dir,True)
        with patch.object(stopped.optimizer,'step',side_effect=AssertionError('must remain stopped')):self.f.quiet_run(stopped)
        bad=copy.deepcopy(b);bad['early_stopping']['no_improvement_count']=0
        save_checkpoint(first.run_dir/'last.ckpt',bad)
        with self.assertRaisesRegex(ValueError,'validation history'):self.tiny_trainer(first.run_dir,True)

    def test_resume_preserves_distinct_raw_best_and_patience_reference(self):
        from organ_relation.metrics import summarize_dice
        scores=iter((.5,.50006,.5001))
        def controlled(records):
            self.assertEqual(len(records),40)
            summary=summarize_dice(records);summary['mean_case_dice']=next(scores)
            return summary
        with patch('organ_relation.training.engine.summarize_dice',side_effect=controlled):
            first=self.tiny_trainer(self.root/'distinct');self.f.quiet_run(first,stop_after=30)
            ck=load_checkpoint(first.run_dir/'last.ckpt',first.identity)
            self.assertEqual(ck['early_stopping']['best_metric'],.50006)
            self.assertEqual(ck['early_stopping']['patience_reference_metric'],.5)
            self.assertEqual(ck['early_stopping']['no_improvement_count'],1)
            best=load_checkpoint(first.run_dir/'best-dev.ckpt',first.identity)
            self.assertEqual(best['development_state']['best_dev']['epoch'],10)
            resumed=self.tiny_trainer(first.run_dir,True)
            self.f.assert_nested_equal(resumed.early_state,ck['early_stopping'])
            self.f.quiet_run(resumed,stop_after=15)
        ck=load_checkpoint(first.run_dir/'last.ckpt',first.identity)
        self.assertEqual(ck['early_stopping']['best_metric'],.5001)
        self.assertEqual(ck['early_stopping']['patience_reference_metric'],.5001)
        self.assertEqual(ck['early_stopping']['no_improvement_count'],0)

    def test_scalar_train_no_observers_epoch_board_and_validation_still_has_soft(self):
        trainer=self.tiny_trainer(self.root/'scalars')
        with patch.object(trainer.criterion,'forward',side_effect=AssertionError('legacy observations in train')):
            self.f.quiet_run(trainer,stop_after=3)
        rows=self.f.rows(trainer.run_dir)
        train=next(r for r in rows if r['phase']=='train')
        epoch=next(r for r in rows if r['phase']=='train_epoch')
        self.assertEqual(set(train['final']),{'segmentation'})
        self.assertNotIn('soft_dice_per_organ',train)
        self.assertNotIn('soft_mean',epoch['epoch_metrics']['final'])
        self.assertIn('hard',epoch['epoch_metrics']['final'])
        events=EventAccumulator(str(trainer.board.directory),size_guidance={'scalars':0}).Reload()
        self.assertEqual([e.step for e in events.Scalars('Loss/Train_Total')],[3])
        self.assertNotIn('Dice/Train_Final_Soft',events.Tags()['scalars'])
        resumed=self.tiny_trainer(trainer.run_dir,True);self.f.quiet_run(resumed,stop_after=12)
        val=next(r for r in self.f.rows(trainer.run_dir) if r['phase']=='internal_dev')
        self.assertEqual(val['metrics']['cases'],40)
        self.assertEqual(len(val['epoch_metrics']['final']['soft_per_organ']),15)

    def test_five_steps_gated_normal_order_optimizer_updates_and_summary(self):
        cfg=config(True);seed_all(cfg['training']['seed'])
        model=MonaiRelationUNet(cfg['model'],cfg['input_processing'],relation_enabled=True,relation_config=cfg['relation'])
        criterion=MonaiRelationLoss(cfg['loss'],**cfg['coarse_supervision'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0,foreach=False,fused=False)
        class Pool(Cases):
            def __len__(self):return 160
            def __getitem__(self,index):return super().__getitem__(index%2)
        ids=[f'synthetic_{i}' for i in range(160)];stream=io.StringIO()
        identity=dict(mode=cfg['mode'],training=cfg['training'],model=cfg['model'],loss=cfg['loss'],
            relation=cfg['relation'],relation_enabled=True,coarse_supervision=cfg['coarse_supervision'],
            optimizer=cfg['optimizer'],preprocessing='synthetic',provenance='test',data={'manifest_hash':'synthetic'})
        trainer=Trainer(model,criterion,optimizer,Pool(),ids,device='cpu',options=cfg['training'],identity=identity,
            run_dir=self.root/'five',validation_dataset=Pool(),validation_case_ids=ids,tensorboard=True,
            console=TrainingConsole(stream=stream))
        with (patch.object(criterion,'forward',side_effect=AssertionError('legacy full observers')),
             patch.object(criterion.branch.dice_per_class,'forward',side_effect=AssertionError('soft Dice observer')),
             patch.object(criterion,'align_coarse',wraps=criterion.align_coarse) as resize,
             patch.object(optimizer,'step',wraps=optimizer.step) as steps):
            self.f.quiet_run(trainer,stop_after=5)
        self.assertEqual(steps.call_count,5);self.assertEqual(resize.call_count,5)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        rows=self.f.rows(trainer.run_dir);train=[r for r in rows if r['phase']=='train']
        expected=torch.randperm(160,generator=torch.Generator().manual_seed(cfg['training']['seed'])).tolist()[:5]
        self.assertEqual([r['case_id'] for r in train],[ids[i] for i in expected])
        self.assertEqual(len(set(r['case_id'] for r in train)),5)
        self.assertEqual(stream.getvalue().count('[Short step]'),5)
        self.assertIn('[Short run summary]',stream.getvalue())
        summary=rows[-1];self.assertEqual(summary['phase'],'short_run_summary')
        self.assertEqual(summary['optimizer_steps'],5)
        self.assertEqual(summary['steps_2_to_5_mean_seconds'],statistics.mean(r['step_seconds'] for r in train[1:]))
        self.assertEqual(summary['steps_2_to_5_median_seconds'],statistics.median(r['step_seconds'] for r in train[1:]))
        self.assertNotEqual(train[0]['relation_scale']['gamma'],train[-1]['gamma_after_step'])
        self.assertEqual(trainer.total,80000)
        ck=load_checkpoint(trainer.run_dir/'last.ckpt',trainer.identity)
        self.assertEqual(ck['progress']['global_step'],5)
        self.assertEqual(ck['progress']['epoch'],0)
        self.assertEqual(len(ck['development_state']['epoch_cases']),5)


if __name__=='__main__':unittest.main()
