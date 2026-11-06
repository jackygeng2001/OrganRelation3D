"""CPU equivalence and resume acceptance for the explicit Conv3d backend."""
import contextlib
import copy
import importlib.util
import io
import json
import unittest

import torch
from torch.nn import functional as F

import test_training as fixtures
from test_brain_tumor import Cases, config
from organ_relation.losses import RegionJointLoss
from organ_relation.models.brain_tumor import BrainTumorConfig, BrainTumorSegmentor
from organ_relation.models.depth_sliced_conv import (
    DEPTH_CHUNK, DepthSlicedConv3d, _DepthSlicedConv, enable_depth_sliced_convolutions)
from organ_relation.training.engine import Trainer
from organ_relation.training.state import seed_all, load_checkpoint


class DepthSlicedTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.TrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_conv_output_and_all_gradients_across_geometry_layout_and_bias(self):
        torch.manual_seed(31)
        geometries = [((1,1,1),(1,1,1),(1,1,1)), ((2,2,2),(1,1,1),(1,1,1)),
                      ((1,1,1),(2,2,2),(2,2,2)), ((2,1,2),(0,1,2),(1,1,2))]
        for dtype in (torch.float32, torch.float64):
            for stride, padding, dilation in geometries:
                for layout in (torch.contiguous_format, torch.channels_last_3d):
                    for affine in (False, True):
                        x = torch.randn(2,4,7,9,11,dtype=dtype).contiguous(memory_format=layout).requires_grad_()
                        w = torch.randn(6,4,3,3,3,dtype=dtype).contiguous(memory_format=layout).requires_grad_()
                        bias = torch.randn(6,dtype=dtype,requires_grad=True) if affine else None
                        params = (x,w,bias) if affine else (x,w)
                        old = F.conv3d(x,w,bias,stride=stride,padding=padding,dilation=dilation)
                        new = _DepthSlicedConv.apply(x,w,bias,stride,padding,dilation,2)
                        grad = torch.randn_like(old)
                        a, b = torch.autograd.grad(old,params,grad), torch.autograd.grad(new,params,grad)
                        tol = dict(rtol=1e-4,atol=3e-4) if dtype==torch.float32 else dict(rtol=1e-10,atol=1e-10)
                        torch.testing.assert_close(old,new,**tol)
                        for u,v in zip(a,b):
                            torch.testing.assert_close(u,v,**tol)

    def test_boundary_singleton_and_frozen_parameters(self):
        for shape in ((1,2,1,1,1),(1,2,3,4,5)):
            for needs in ((True,False,False),(False,True,False),(False,False,True)):
                x = torch.randn(shape,dtype=torch.double,requires_grad=needs[0])
                w = torch.randn(3,2,3,3,3,dtype=torch.double,requires_grad=needs[1])
                bias = torch.randn(3,dtype=torch.double,requires_grad=needs[2])
                old = F.conv3d(x,w,bias,padding=2,dilation=2)
                new = _DepthSlicedConv.apply(x,w,bias,(1,1,1),(2,2,2),(2,2,2),2)
                params = tuple(v for v in (x,w,bias) if v.requires_grad)
                torch.testing.assert_close(old,new,rtol=1e-10,atol=1e-10)
                for u,v in zip(torch.autograd.grad(old.sum(),params),torch.autograd.grad(new.sum(),params)):
                    torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)

    def test_gradcheck_and_double_backward(self):
        x = torch.randn(1,2,3,3,3,dtype=torch.double,requires_grad=True)
        w = torch.randn(2,2,3,3,3,dtype=torch.double,requires_grad=True)
        b = torch.randn(2,dtype=torch.double,requires_grad=True)
        call = lambda x,w,b: _DepthSlicedConv.apply(x,w,b,(1,1,1),(1,1,1),(1,1,1),2)
        self.assertTrue(torch.autograd.gradcheck(call,(x,w,b)))
        self.assertTrue(torch.autograd.gradgradcheck(call,(x,w,b)))

    def test_conversion_preserves_parameter_objects_rng_keys_and_mode(self):
        model = torch.nn.Sequential(torch.nn.Conv3d(2,4,3,padding=1),torch.nn.Conv3d(4,3,1)).eval()
        params = dict(model.named_parameters())
        state = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        enable_depth_sliced_convolutions(model)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(state),list(model.state_dict()))
        for name,param in model.named_parameters():
            self.assertIs(param,params[name])
            torch.testing.assert_close(param,state[name],rtol=0,atol=0)
        self.assertFalse(model[0].training)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_bounded_conv2d_batches_and_only_original_tensors_saved(self):
        from unittest.mock import patch
        x = torch.randn(2,3,13,5,7,requires_grad=True)
        w = torch.randn(4,3,3,3,3,requires_grad=True)
        sizes, saved = [], []
        original = F.conv2d
        def observe(image,*args,**kwargs):
            sizes.append(image.shape[0])
            self.assertTrue(image.is_contiguous())
            return original(image,*args,**kwargs)
        def pack(tensor):
            saved.append(tensor)
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t), patch(
                'organ_relation.models.depth_sliced_conv.F.conv2d',side_effect=observe):
            output = _DepthSlicedConv.apply(x,w,None,(1,1,1),(1,1,1),(1,1,1),DEPTH_CHUNK)
        self.assertLessEqual(max(sizes),x.shape[0]*DEPTH_CHUNK)
        self.assertEqual(len(saved),2)
        self.assertIs(saved[0],x)
        self.assertIs(saved[1],w)
        self.assertTrue(output.is_contiguous(memory_format=torch.channels_last_3d))

    def test_pointwise_linear_and_rejection_of_unsupported_modes(self):
        native = torch.nn.Conv3d(4,3,1).double()
        fast = DepthSlicedConv3d.from_native(native)
        x = torch.randn(2,4,3,5,7,dtype=torch.double,requires_grad=True)
        a,b = native(x),fast(x)
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
        for u,v in zip(torch.autograd.grad(a.sum(),(x,*native.parameters())),
                       torch.autograd.grad(b.sum(),(x,*fast.parameters()))):
            torch.testing.assert_close(u,v,rtol=1e-12,atol=1e-12)
        for module in (torch.nn.Conv3d(4,4,3,groups=2),
                       torch.nn.Conv3d(4,4,3,padding_mode='reflect')):
            with self.assertRaises(ValueError):
                DepthSlicedConv3d.from_native(module)
        with self.assertRaises(ValueError):
            fast(x.half())
        with self.assertRaises(ValueError):
            DepthSlicedConv3d(4,4,3,groups=2)(x.float())

    def test_concatenated_inputs_outputs_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for layout in (torch.contiguous_format, torch.channels_last_3d):
                for stride, dilation in ((1,1), (2,1), (1,2)):
                    conv = DepthSlicedConv3d(5,4,3,stride=stride,padding=dilation,dilation=dilation).to(dtype)
                    a = torch.randn(2,2,7,5,9,dtype=dtype).contiguous(memory_format=layout).requires_grad_()
                    b = torch.randn(2,3,7,5,9,dtype=dtype).contiguous(memory_format=layout).requires_grad_()
                    reference = F.conv3d(torch.cat((a,b),1),conv.weight,conv.bias,
                                         stride=stride,padding=dilation,dilation=dilation)
                    result = conv.forward_concatenated(a,b)
                    upstream = torch.randn_like(reference)
                    params = (a,b,*conv.parameters())
                    tol = dict(rtol=1e-4,atol=3e-4) if dtype==torch.float32 else dict(rtol=1e-10,atol=1e-10)
                    torch.testing.assert_close(result,reference,**tol)
                    for u,v in zip(torch.autograd.grad(result,params,upstream),
                                   torch.autograd.grad(reference,params,upstream)):
                        torch.testing.assert_close(u,v,**tol)
        with self.assertRaises(ValueError):
            conv.forward_concatenated(a,b[:,:,:-1])

    def test_concatenated_saved_storage_and_bounded_materialization(self):
        from unittest.mock import patch
        a = torch.randn(1,2,13,5,7,requires_grad=True)
        b = torch.randn(1,3,13,5,7,requires_grad=True)
        conv = DepthSlicedConv3d(5,4,3,padding=1)
        saved, sizes = [], []
        original = torch.cat
        def cat(tensors,dim=0):
            self.assertTrue(all(t.ndim==4 and t.shape[0]<=DEPTH_CHUNK for t in tensors))
            sizes.append(tensors[0].shape[0])
            return original(tensors,dim)
        def pack(tensor):
            saved.append(tensor)
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t), patch(
                'organ_relation.models.depth_sliced_conv.torch.cat',side_effect=cat):
            output = conv.forward_concatenated(a,b)
            output.sum().backward()
        self.assertTrue(sizes)
        self.assertEqual(len(saved),3)
        self.assertIs(saved[0],a)
        self.assertIs(saved[1],conv.weight)
        self.assertIs(saved[2],b)

    def test_decoder_uses_two_inputs_through_normal_module_call(self):
        model = BrainTumorSegmentor(BrainTumorConfig(**config()['model']),convolution_backend='depth_sliced')
        observed, handles = [], []
        for block in model.decoder.fuse:
            handles.append(block[0][0].register_forward_pre_hook(
                lambda module,args: observed.append(tuple(tuple(x.shape) for x in args))))
        try:
            model(torch.randn(1,4,9,11,13)).final_logits.sum().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(observed),3)
        self.assertTrue(all(len(shapes)==2 and shapes[0]==shapes[1] for shapes in observed))
        self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))

    def test_concatenated_double_backward_and_frozen_inputs(self):
        a = torch.randn(1,1,2,3,3,dtype=torch.double,requires_grad=True)
        b = torch.randn_like(a,requires_grad=True)
        w = torch.randn(2,2,3,3,3,dtype=torch.double,requires_grad=True)
        bias = torch.randn(2,dtype=torch.double,requires_grad=True)
        call = lambda a,b,w,bias: _DepthSlicedConv.apply(a,w,bias,(1,1,1),(1,1,1),(1,1,1),2,b)
        self.assertTrue(torch.autograd.gradcheck(call,(a,b,w,bias)))
        self.assertTrue(torch.autograd.gradgradcheck(call,(a,b,w,bias)))
        for mask in ((True,False,False,False),(False,True,False,False),(False,False,True,True)):
            values = tuple(t.detach().requires_grad_(need) for t,need in zip((a,b,w,bias),mask))
            inputs = tuple(t for t in values if t.requires_grad)
            ref = F.conv3d(torch.cat(values[:2],1),values[2],values[3],padding=1)
            actual = call(*values)
            for u,v in zip(torch.autograd.grad(ref.sum(),inputs),torch.autograd.grad(actual.sum(),inputs)):
                torch.testing.assert_close(u,v,rtol=1e-10,atol=1e-10)

    def test_full_model_initialization_outputs_loss_gradients_and_no_hooks(self):
        cfg = config()
        for dtype in (torch.float32,torch.float64):
            seed_all(17)
            old = BrainTumorSegmentor(BrainTumorConfig(**cfg['model'])).to(dtype=dtype,memory_format=torch.channels_last_3d)
            rng = torch.get_rng_state().clone()
            seed_all(17)
            new = BrainTumorSegmentor(BrainTumorConfig(**cfg['model']),convolution_backend='depth_sliced').to(dtype=dtype,memory_format=torch.channels_last_3d)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(list(old.state_dict()),list(new.state_dict()))
            for u,v in zip(old.parameters(),new.parameters()):
                torch.testing.assert_close(u,v,rtol=0,atol=0)
            x = torch.randn(1,4,9,11,13,dtype=dtype).contiguous(memory_format=torch.channels_last_3d).requires_grad_()
            z = x.detach().clone().requires_grad_()
            y = torch.arange(9*11*13).reshape(1,9,11,13)%4
            a,b = old(x),new(z)
            criterion = RegionJointLoss(**cfg['loss'])
            la,lb = criterion(*a,y).total,criterion(*b,y).total
            la.backward();lb.backward()
            tol = dict(rtol=3e-4,atol=3e-6) if dtype==torch.float32 else dict(rtol=1e-9,atol=1e-11)
            for u,v in zip(a,b):
                torch.testing.assert_close(u,v,**tol)
            torch.testing.assert_close(la,lb,**tol)
            torch.testing.assert_close(x.grad,z.grad,**tol)
            # FP32 reassociation can move an activation across zero; use an
            # absolute gradient tolerance, retaining strict FP64 equivalence.
            grad_tol = dict(rtol=3e-4,atol=1e-5) if dtype==torch.float32 else tol
            for (name,u),(other,v) in zip(old.named_parameters(),new.named_parameters()):
                self.assertEqual(name,other)
                torch.testing.assert_close(u.grad,v.grad,**grad_tol,msg=lambda message: str(dtype)+" "+name+" "+message)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in new.modules()))

    def test_cli_explicit_backend_provenance_and_native_default(self):
        spec = importlib.util.spec_from_file_location('conv_cli_test',fixtures.ROOT/'scripts/brats.py')
        cli = importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
        for backend in ('native','depth_sliced'):
            out = io.StringIO()
            args = ['check','--shape','5','7','9']
            if backend=='depth_sliced':
                args += ['--convolution-backend',backend]
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.main(args),0)
            result = json.loads(out.getvalue())
            self.assertEqual(result['convolution_backend'],backend)
            self.assertEqual(result['convolution_depth_chunk'],DEPTH_CHUNK if backend=='depth_sliced' else None)
            self.assertEqual(result['optimizer_steps'],0)
            self.assertEqual(result['missing_gradients'],[])
            self.assertEqual(result['nonfinite_gradients'],[])

    def test_resume_same_trajectory_and_backend_change_rejected(self):
        cfg = config()
        def trainer(path,resume=False):
            seed_all(171)
            model = BrainTumorSegmentor(BrainTumorConfig(**cfg['model']),convolution_backend='depth_sliced')
            options = dict(cfg['training'],max_steps=2,validation_every=0,diagnostics_every=1,checkpoint_every=1)
            identity = dict(mode='brats_regions',model=cfg['model'],loss=cfg['loss'],threshold=.5,training=options,
                            runtime=dict(convolution_backend='depth_sliced',convolution_depth_chunk=DEPTH_CHUNK),
                            provenance='test',data={'manifest_hash':'synthetic'})
            optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,foreach=False,fused=False)
            return Trainer(model,RegionJointLoss(**cfg['loss']),optimizer,Cases(),['one','two'],
                           device='cpu',options=options,identity=identity,run_dir=path,
                           resume=path/'last.ckpt' if resume else None)
        whole = trainer(self.fixture.root/'whole');self.fixture.quiet_run(whole)
        part = trainer(self.fixture.root/'part');self.fixture.quiet_run(part,stop_after=1)
        resumed = trainer(part.run_dir,True);self.fixture.quiet_run(resumed)
        a = load_checkpoint(whole.run_dir/'last.ckpt',whole.identity)
        b = load_checkpoint(resumed.run_dir/'last.ckpt',resumed.identity)
        for key in ('model','optimizer','rng','progress','sampler_generator','loader_generator'):
            self.fixture.assert_nested_equal(a[key],b[key])
        changed = copy.deepcopy(resumed.identity)
        changed['runtime']['convolution_backend']='native'
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            load_checkpoint(resumed.run_dir/'last.ckpt',changed)



class InterleavedDilationTests(unittest.TestCase):
    """In-plane dilated 'same' kernels run as interleaved ordinary convolutions."""

    def observe(self, run):
        from unittest.mock import patch
        dilations = []
        original = F.conv2d
        def conv2d(image, weight, *args, **kwargs):
            dilations.append(tuple(kwargs.get('dilation', (1, 1))) if 'dilation' in kwargs else (1, 1))
            return original(image, weight, *args, **kwargs)
        with patch('organ_relation.models.depth_sliced_conv.F.conv2d', side_effect=conv2d):
            result = run()
        return result, dilations

    def compare(self, shape, stride, padding, dilation):
        g = torch.Generator().manual_seed(7)
        x = torch.randn(*shape, generator=g, dtype=torch.float64).contiguous(
            memory_format=torch.channels_last_3d).requires_grad_()
        w = torch.randn(4, shape[1], 3, 3, 3, generator=g, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, generator=g, dtype=torch.float64, requires_grad=True)
        old = F.conv3d(x, w, bias, stride=stride, padding=padding, dilation=dilation)
        new, dilations = self.observe(lambda: _DepthSlicedConv.apply(x, w, bias, stride, padding, dilation, 2))
        grad = torch.randn(old.shape, generator=g, dtype=torch.float64)
        torch.testing.assert_close(new, old, rtol=1e-10, atol=1e-10)
        for u, v in zip(torch.autograd.grad(new, (x, w, bias), grad), torch.autograd.grad(old, (x, w, bias), grad)):
            torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-10)
        return dilations

    def test_dilated_same_kernels_never_reach_the_dilated_solver(self):
        for shape, dilation in (((1, 3, 6, 9, 11), (2, 2, 2)), ((2, 3, 5, 8, 7), (1, 1, 2)),
                                ((1, 3, 5, 7, 10), (1, 2, 1)), ((1, 2, 7, 10, 11), (3, 3, 3)),
                                ((1, 2, 3, 1, 3), (2, 2, 2))):
            with self.subTest(shape=shape, dilation=dilation):
                dilations = self.compare(shape, (1, 1, 1), dilation, dilation)
                self.assertTrue(dilations)
                self.assertEqual(set(dilations), {(1, 1)})

    def test_other_geometries_keep_the_original_call(self):
        # Not 'same' (padding != dilation) or strided in-plane: one plain dilated call per plane batch.
        for stride, padding, dilation in (((1, 1, 1), (1, 1, 1), (2, 2, 2)), ((1, 2, 2), (2, 2, 2), (2, 2, 2))):
            with self.subTest(stride=stride, padding=padding):
                dilations = self.compare((1, 3, 6, 9, 11), stride, padding, dilation)
                self.assertEqual(set(dilations), {dilation[1:]})

    def test_module_and_concatenated_inputs_use_interleaving(self):
        conv = DepthSlicedConv3d(5, 4, 3, padding=2, dilation=2).double()
        a = torch.randn(1, 2, 6, 7, 9, dtype=torch.float64, requires_grad=True)
        b = torch.randn(1, 3, 6, 7, 9, dtype=torch.float64, requires_grad=True)
        reference = F.conv3d(torch.cat((a, b), 1), conv.weight, conv.bias, padding=2, dilation=2)
        out, dilations = self.observe(lambda: conv(a, b))
        torch.testing.assert_close(out, reference, rtol=1e-10, atol=1e-10)
        for u, v in zip(torch.autograd.grad(out.sum(), (a, b, conv.weight)),
                        torch.autograd.grad(reference.sum(), (a, b, conv.weight))):
            torch.testing.assert_close(u, v, rtol=1e-10, atol=1e-10)
        self.assertEqual(set(dilations), {(1, 1)})

if __name__ == '__main__':
    unittest.main()
