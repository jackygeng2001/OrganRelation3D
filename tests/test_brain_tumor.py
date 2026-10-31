"""Revised-method formula, geometry, data and resume acceptance on CPU only."""
import copy
import gzip
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import test_training as fixtures
from organ_relation.models.brain_tumor import BrainTumorConfig,BrainTumorSegmentor,DualReceptiveField,scale_descriptor
from organ_relation.models.space_to_node import SpaceToNode,OrganNodes
from organ_relation.models.dynamic_relation import DynamicRelation
from organ_relation.models.node_to_space import NodeToSpace
from organ_relation.losses import RegionJointLoss,region_targets
from organ_relation.data.brats import BraTSDataset,index_cases,canonical_labels,validate_partition,MODALITIES
from organ_relation.data.full_scan import FullScanSample
from organ_relation.training.brain_tumor import region_dice
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all,digest,load_checkpoint


def config():
    return json.loads((fixtures.ROOT/'configs/brats_regions.json').read_text())


class Cases(torch.utils.data.Dataset):
    def __len__(self):return 2
    def __getitem__(self,index):
        shape=(5+index,7,9)
        return FullScanSample(torch.randn(4,*shape),(torch.arange(np.prod(shape)).reshape(shape)%4).long(),{})


class BrainTumorTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.TrainingTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cfg=config()

    def model(self):return BrainTumorSegmentor(BrainTumorConfig(**self.cfg['model']))
    def criterion(self):return RegionJointLoss(**self.cfg['loss'])

    def test_dual_branch_formula_and_gn_dilation(self):
        module=DualReceptiveField(64,bias=True).double()
        x=torch.randn(1,64,3,4,5,dtype=torch.double,requires_grad=True)
        a,b=module.local(x),module.context(x)
        logits=module.gate(torch.cat((a,b,(a-b).abs()),1))
        w=logits.softmax(1)
        expected=x+module.projection(w[:,:1]*a+w[:,1:]*b)
        torch.testing.assert_close(module(x),expected,rtol=0,atol=0)
        self.assertEqual(module.context[0].dilation,(2,2,2))
        self.assertEqual(module.context[0].padding,(2,2,2))
        self.assertEqual(module.local[1].num_groups,8)
        self.assertEqual(module.gate[0].weight.shape,(16,192,1,1,1))

    def test_three_overlapping_node_pool_no_background(self):
        feature=torch.randn(1,2,1,2,3,dtype=torch.double,requires_grad=True)
        p=torch.full((1,3,1,2,3),.8,dtype=torch.double,requires_grad=True)
        nodes=SpaceToNode(epsilon=1e-6,node_count=3,include_background=False)(feature,p)
        self.assertEqual(nodes.z0.shape,(1,3,2))
        torch.testing.assert_close(nodes.mass,torch.full((1,3,1),4.8,dtype=torch.double))
        expected=torch.einsum('bin,bcn->bic',p.flatten(2),feature.flatten(2))/(nodes.mass+1e-6)
        torch.testing.assert_close(nodes.z0,expected)
        (nodes.z0.sum()+nodes.centroid.sum()+nodes.size.sum()+nodes.confidence.sum()).backward()
        self.assertGreater(p.grad.abs().sum(),0)

    def test_six_directed_edges_last_gate_before_update_exactly_K_calls(self):
        module=DynamicRelation(4,relation_channels=5,rounds=2,node_count=3).double()
        inputs=[torch.randn(1,3,n,dtype=torch.double,requires_grad=True) for n in (4,3,1,1)]
        with patch.object(module,'_edge_weights',wraps=module._edge_weights) as calls:
            result=module.forward_with_last_edges(*inputs)
            self.assertEqual(calls.call_count,2)
            pre=calls.call_args.args[0]
        torch.testing.assert_close(result.last_alpha,module._edge_weights(pre,*inputs[1:]),rtol=0,atol=0)
        self.assertFalse(torch.equal(pre,result.zK))
        self.assertEqual(int((result.last_alpha!=0).sum()),6)
        self.assertTrue((result.last_alpha.diagonal(dim1=1,dim2=2)==0).all())
        torch.testing.assert_close(module._aggregate_messages(inputs[0],result.last_alpha),
            torch.stack([sum(result.last_alpha[:,i,j:j+1]*module.W_m(inputs[0])[:,i] for i in range(3) if i!=j) for j in range(3)],1))
        (result.zK.sum()+result.last_alpha.sum()).backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in inputs))

    def test_scale_descriptor_incoming_order_and_scale_softmax(self):
        z=torch.randn(1,3,256,requires_grad=True)
        nodes=OrganNodes(z,torch.ones(1,3,1),torch.randn(1,3,3),torch.randn(1,3,1),torch.randn(1,3,1))
        alpha=torch.tensor([[[0.,12,13],[21,0,23],[31,32,0]]],requires_grad=True)
        xi=scale_descriptor(z,nodes,alpha)
        self.assertEqual(xi.shape,(1,3,266))
        torch.testing.assert_close(xi[0,:,261:263],torch.tensor([[21.,31],[12,32],[13,23]]))
        torch.testing.assert_close(xi[0,:,263:],torch.eye(3))
        model=self.model();weights=model.decoder.scale_mlp(xi).softmax(-1)
        torch.testing.assert_close(weights.sum(-1),torch.ones(1,3))
        weights[0,0,0].backward()
        self.assertGreater(alpha.grad.abs().sum(),0)
        self.assertTrue(all(p.bias is None for p in model.decoder.skip_projection))
        torch.testing.assert_close(model.decoder.eta,torch.full((3,),1e-3))

    def test_multiscale_query_weighted_content_and_gradients(self):
        module=NodeToSpace(6,key_channels=8,node_count=3,attention_channels=4,content_channels=5).double()
        x=torch.randn(1,8,2,3,4,dtype=torch.double,requires_grad=True)
        z=torch.randn(1,3,6,dtype=torch.double,requires_grad=True)
        weight=torch.rand(1,3,1,dtype=torch.double,requires_grad=True)
        result=module.forward_with_scale_weights(x,z,weight,return_diagnostics=True)
        expected=sum((weight[:,i:i+1]*result.A[:,i:i+1]).transpose(1,2)*result.V[:,i:i+1] for i in range(3))
        torch.testing.assert_close(result.G.flatten(2).transpose(1,2),expected)
        result.G.square().mean().backward()
        self.assertTrue(all(t.grad is not None for t in (x,z,weight)))

    def test_complete_document_architecture_odd_shapes_all_gradients_image_only(self):
        model=self.model()
        x=torch.randn(1,4,9,11,13,requires_grad=True)
        out=model.forward_with_diagnostics(x)
        self.assertEqual([tuple(t.shape[1:]) for t in out.features],[(32,9,11,13),(64,5,6,7),(128,3,3,4),(256,2,2,2)])
        self.assertEqual(out.output.final_logits.shape,(1,3,9,11,13))
        self.assertEqual(out.scale_weights.shape,(1,3,3))
        label=torch.arange(9*11*13).reshape(1,9,11,13)%4
        self.criterion()(*out.output,label).total.backward()
        for name,p in model.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertGreater(model.decoder.eta.grad.abs().sum(),0)
        self.assertGreater(model.fusion.relation_scale.grad.abs().sum(),0)
        with self.assertRaises(TypeError):model(x,label=label)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_exact_multilabel_probability_bce_dice_batch_and_interpolation(self):
        criterion=self.criterion()
        label=(torch.arange(48).reshape(2,2,3,4)%4).long()
        target=region_targets(label).double()
        torch.testing.assert_close(target[0,:,0,0,3],torch.ones(3,dtype=torch.double))
        coarse=torch.randn(2,3,1,2,2,dtype=torch.double,requires_grad=True)
        final=torch.randn(2,3,2,3,4,dtype=torch.double,requires_grad=True)
        eps=criterion.epsilon
        def branch(logits):
            s=logits.sigmoid()
            bce=-(target*(s+eps).log()+(1-target)*(1-s+eps).log()).mean((1,2,3,4))
            dice=(2*(s*target).sum((2,3,4))+eps)/(s.sum((2,3,4))+target.sum((2,3,4))+eps)
            return bce+1-dice.mean(1)
        aligned=F.interpolate(coarse,size=label.shape[1:],mode='trilinear',align_corners=False)
        expected=(branch(final)+.5*branch(aligned)).mean()
        actual=criterion(coarse,final,label)
        torch.testing.assert_close(actual.total,expected)
        for a,b in zip(torch.autograd.grad(actual.total,(coarse,final),retain_graph=True),torch.autograd.grad(expected,(coarse,final))):
            torch.testing.assert_close(a,b)
        self.assertEqual(actual.final.dice_per_class.shape,(2,3))
        with self.assertRaises(ValueError):region_targets(torch.full((1,2,3,4),4,dtype=torch.long))

    def test_independent_threshold_keeps_overlap_and_reports_violations(self):
        label=torch.tensor([[[[0,1,2,3]]]])
        logits=torch.where(region_targets(label),10.,-10.)
        score=region_dice(logits,label,.5)
        self.assertEqual(score['mean_dice'],1)
        logits[0,2,0,0,0]=10
        score=region_dice(logits,label,.5)
        self.assertEqual(score['hierarchy_violations']['ET_outside_TC'],1)
        self.assertEqual(score['organs'][2]['predicted_voxels'],2)

    def make_source(self):
        source=self.fixture.root/'data';source.mkdir()
        grid=np.arange(60,dtype=np.float32).reshape(3,4,5)
        for case in ('BraTS-GLI-00001-000','BraTS-GLI-00001-001','BraTS-GLI-00002-000'):
            folder=source/case;folder.mkdir()
            for index,role in enumerate((*MODALITIES,'seg')):
                arr=(grid%4).astype(np.int16) if role=='seg' else grid*(index+1)
                nib.save(nib.Nifti1Image(arr,np.eye(4)),folder/(case+'-'+role+'.nii.gz'))
        return source

    def test_full_grid_mri_zscore_zip_equivalence_and_label_semantics(self):
        source=self.make_source();manifest=index_cases(source,cohort='brats2023_gli')
        dataset=BraTSDataset(manifest,**self.cfg['preprocessing']);sample=dataset[0]
        self.assertEqual(sample.image.shape,(4,5,4,3));self.assertEqual(sample.label.shape,(5,4,3))
        for channel in sample.image:
            self.assertEqual(channel[0,0,0],0)
            self.assertAlmostEqual(channel.flatten()[1:].mean().item(),0,places=6)
        archive=self.fixture.root/'data.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for p in source.rglob('*.nii.gz'):z.write(p,p.relative_to(source).as_posix())
        other=BraTSDataset(index_cases(archive,cohort='brats2023_gli'),**self.cfg['preprocessing'])[0]
        torch.testing.assert_close(sample.image,other.image,rtol=0,atol=0)
        self.assertTrue(torch.equal(sample.label,other.label))
        with self.assertRaisesRegex(ValueError,'2024'):
            BraTSDataset(index_cases(archive,cohort='brats2024_gli_post'),**self.cfg['preprocessing'])
        with self.assertRaises(ValueError):canonical_labels(np.array([0,1,2,4]),encoding='0123')
        self.assertEqual(canonical_labels(np.array([4]),encoding='0124_preoperative').tolist(),[3])
        record=manifest['records'][0];path=source/record['files']['t1c']['path']
        nib.save(nib.Nifti1Image(np.ones((3,4,5),dtype=np.float32),np.diag([2.,1,1,1])),path)
        with self.assertRaisesRegex(ValueError,'grids differ'):dataset[0]

    def test_patient_group_split_rejects_timepoint_leakage(self):
        source=self.make_source();manifest=index_cases(source,cohort='brats2023_gli')
        ids=[r['case_id'] for r in manifest['records']]
        split=dict(manifest_hash=digest(manifest),train=ids[:1],dev=ids[1:])
        with self.assertRaisesRegex(ValueError,'leakage'):validate_partition(manifest,split)
        split.update(train=ids[:2],dev=ids[2:]);self.assertEqual(len(validate_partition(manifest,split)[0]),2)
        split['manifest_hash']='bad'
        with self.assertRaisesRegex(ValueError,'hash'):validate_partition(manifest,split)

    def test_known_postoperative_archive_cannot_be_declared_preoperative(self):
        source=self.make_source()
        archive=self.fixture.root/'BraTS2024-BraTS-GLI-TrainingData.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for p in source.rglob('*.nii.gz'):
                z.write(p,p.relative_to(source).as_posix())
        with self.assertRaisesRegex(ValueError,'conflicts with declared cohort'):
            index_cases(archive,cohort='brats2023_gli')
        manifest=index_cases(archive,cohort='brats2024_gli_post')
        self.assertEqual(len(manifest['records']),3)
        with self.assertRaisesRegex(ValueError,'separately agreed protocol'):
            BraTSDataset(manifest,**self.cfg['preprocessing'])

    def test_cpu_channels_last_outputs_and_gradients_match(self):
        torch.manual_seed(34)
        ordinary=self.model()
        channels_last=copy.deepcopy(ordinary).to(memory_format=torch.channels_last_3d)
        x=torch.randn(1,4,5,7,9,requires_grad=True)
        y=x.detach().clone().contiguous(memory_format=torch.channels_last_3d).requires_grad_()
        label=torch.arange(5*7*9).reshape(1,5,7,9)%4
        a,b=ordinary(x),channels_last(y)
        for v,w in zip(a,b):torch.testing.assert_close(v,w,rtol=1e-4,atol=1e-5)
        self.criterion()(*a,label).total.backward()
        self.criterion()(*b,label).total.backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-4,atol=1e-6)
        for p,q in zip(ordinary.parameters(),channels_last.parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)

    def test_cli_cpu_no_update_preflight_and_explicit_source_guards(self):
        spec=importlib.util.spec_from_file_location('brats_cli_test',fixtures.ROOT/'scripts/brats.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        import contextlib
        stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(module.main(['check','--shape','5','7','9']),0)
        result=json.loads(stdout.getvalue())
        self.assertEqual(result['parameter_count'],8330296)
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual(result['missing_gradients'],[])
        self.assertEqual(result['nonfinite_gradients'],[])
        self.assertIsNone(result['memory']['peak_allocated_bytes'])
        self.assertNotIn('phases',result)  # Default remains the original unprofiled check.
        for args in (['check','--run-dir',str(self.fixture.root/'forbidden')],
                     ['check','--source','missing'],['train'],['check','--shape','100','100','100']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(args),2)
        self.assertFalse((self.fixture.root/'forbidden').exists())

    def profile_cli(self):
        spec=importlib.util.spec_from_file_location('brats_profile_test',fixtures.ROOT/'scripts/brats.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_profile_all_phases_preserves_loss_gradients_rng_and_cleans_hooks(self):
        module=self.profile_cli()
        model=self.model().to(memory_format=torch.channels_last_3d)
        reference=copy.deepcopy(model)
        x=torch.randn(1,4,5,7,9).contiguous(memory_format=torch.channels_last_3d).requires_grad_()
        y=x.detach().clone().requires_grad_()
        label=torch.arange(5*7*9).reshape(1,5,7,9)%4
        rng=torch.get_rng_state().clone()
        result=module.profile_check(model,self.criterion(),x,label,'cpu')
        expected=self.criterion()(*reference(y),label).total
        expected.backward()
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['loss'],expected.item())
        self.assertEqual(result['optimizer_steps'],0)
        self.assertEqual([p['phase'] for p in result['phases']],['forward','loss','backward'])
        self.assertTrue(result['image']['channels_last_3d'])
        self.assertIsNone(result['peak_allocated_bytes'])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        json.dumps(result,allow_nan=False)

    def test_profile_oom_each_phase_reports_operator_without_retry_or_hooks(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        module=self.profile_cli()
        for phase,operation in (('forward','aten.cat.default'),('loss','aten.log.default'),
                                ('backward','aten.convolution_backward.default')):
            with self.subTest(phase=phase):
                model=self.model();criterion=self.criterion()
                class FailOperator(TorchDispatchMode):
                    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                        match=str(func)==operation
                        if operation=='aten.cat.default':
                            match=match and len(args[0])==2 and args[0][0].shape[1]==32
                        if match:raise torch.OutOfMemoryError('synthetic Tried to allocate 2.13 GiB')
                        return func(*args,**(kwargs or {}))
                with FailOperator(),patch.object(model,'forward',wraps=model.forward) as calls:
                    result=module.profile_check(model,criterion,torch.randn(1,4,5,7,9),
                                               torch.zeros(1,5,7,9,dtype=torch.long),'cpu')
                self.assertEqual(calls.call_count,1)
                self.assertEqual(result['status'],'failed')
                self.assertEqual(result['failure']['phase'],phase)
                self.assertTrue(result['failure']['oom'])
                active=result['failure']['active_scopes']
                self.assertEqual(active[-1]['name'],operation)
                if phase=='forward':
                    self.assertIn('decoder',active[-1]['module_scopes'])
                    self.assertEqual(active[-1]['inputs']['args'][0][0]['shape'],[1,32,5,7,9])
                self.assertEqual(len(result['phases']),('forward','loss','backward').index(phase)+1)
                self.assertTrue(all(p['status']=='passed' for p in result['phases'][:-1]))
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks
                                    for m in (*model.modules(),*criterion.modules())))
                self.assertEqual(result['optimizer_steps'],0)
                json.dumps(result,allow_nan=False)

    def test_profile_cli_is_opt_in_check_only_and_prints_failure_before_exit(self):
        import contextlib
        module=self.profile_cli()
        stdout=io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(module.main(['check','--shape','5','7','9','--profile-memory']),0)
        report=json.loads(stdout.getvalue())
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['memory_format'],'channels_last_3d')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(module.main(['train','--profile-memory']),2)
        for error in (torch.OutOfMemoryError('synthetic loss OOM'),ValueError('synthetic nonfinite loss')):
            stdout=io.StringIO()
            with patch.object(RegionJointLoss,'forward',side_effect=error),\
                 contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(['check','--shape','5','7','9','--profile-memory']),2)
            report=json.loads(stdout.getvalue())
            self.assertEqual(report['status'],'failed')
            self.assertEqual(report['failure']['phase'],'loss')
            self.assertEqual(report['failure']['oom'],isinstance(error,torch.OutOfMemoryError))

    def trainer(self,path,resume=False):
        seed_all(171);model=self.model();criterion=self.criterion()
        options=dict(self.cfg['training'],max_steps=2,validation_every=1,diagnostics_every=1,checkpoint_every=1,validation_role='train_monitor')
        identity=dict(mode='brats_regions',model=self.cfg['model'],loss=self.cfg['loss'],threshold=.5,training=options,
                      preprocessing=self.cfg['preprocessing'],provenance='test',data={'manifest_hash':'synthetic'})
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,foreach=False,fused=False)
        return Trainer(model,criterion,optimizer,Cases(),['one','two'],device='cpu',options=options,identity=identity,
            run_dir=path,validation_dataset=Cases(),validation_case_ids=['one','two'],tensorboard=True,
            resume=path/'last.ckpt' if resume else None)

    def test_shared_engine_resume_state_rng_case_order_and_three_region_logs(self):
        whole=self.trainer(self.fixture.root/'whole');self.fixture.quiet_run(whole)
        first=self.trainer(self.fixture.root/'resumed');self.fixture.quiet_run(first,stop_after=1)
        second=self.trainer(first.run_dir,True);self.fixture.quiet_run(second)
        a=load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b=load_checkpoint(second.run_dir/'last.ckpt',second.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        for x,y in zip(self.fixture.rows(whole.run_dir),self.fixture.rows(second.run_dir)):
            for key in ('phase','global_step','case_id','total_loss','final','coarse','metrics','region_diagnostic_cases'):
                self.assertEqual(x.get(key),y.get(key))
        self.assertFalse(second.board.failed)
        ev=EventAccumulator(str(second.board.directory),size_guidance={'scalars':0}).Reload()
        self.assertEqual([e.step for e in ev.Scalars('Monitor/HardDice/WT')],[1,2])
        self.assertNotIn('MonitorDice/Spleen',ev.Tags()['scalars'])
        changed=copy.deepcopy(second.identity);changed['threshold']=.4
        with self.assertRaisesRegex(ValueError,'identity mismatch'):load_checkpoint(second.run_dir/'last.ckpt',changed)


if __name__=='__main__':unittest.main()
