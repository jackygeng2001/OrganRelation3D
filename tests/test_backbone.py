"""Small synthetic CPU tests. No real CT arrays, graph modules or training."""
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from organ_relation.backbone_config import BackboneConfig
ROOT=Path(__file__).resolve().parents[1]
try:
    import torch
except ModuleNotFoundError as exc:
    if exc.name!='torch':raise
    TORCH_AVAILABLE=False
else:
    TORCH_AVAILABLE=True
    from organ_relation.backbone import Encoder3D,Decoder3D,UNetBackbone3D,resize_to_skip
    from organ_relation.backbone_diagnostics import DeviceMemoryMonitor,parameter_inventory


def micro(**overrides):
    raw=json.loads((ROOT/'configs/backbone_micro.json').read_text(encoding='utf-8'))['model']
    return replace(BackboneConfig(**raw),**overrides)


class BackboneConfigTests(unittest.TestCase):
    def test_explicit_shape_algebra(self):
        self.assertEqual(micro().spatial_pyramid((17,18,19)),((17,18,19),(9,9,10),(5,5,5)))
        c=micro(channels=(2,4,8,16),downsample_strides=((1,2,2),(2,2,2),(2,1,2)))
        self.assertEqual(c.spatial_pyramid((9,19,21)),((9,19,21),(9,10,11),(5,5,6),(3,5,3)))

    def test_config_serialization(self):
        c=micro();self.assertEqual(BackboneConfig(**json.loads(json.dumps(c.to_dict()))),c)

    def test_configuration_failures(self):
        for kwargs in [{'channels':(4,)},{'channels':(4,0,16)},
                       {'downsample_strides':((2,2,2),)},
                       {'downsample_strides':((1,1,1),(2,2,2))},
                       {'downsample_strides':((3,2,2),(2,2,2))},
                       {'normalization':'batch'},{'norm_groups':3},{'norm_eps':0},
                       {'negative_slope':float('nan')},{'align_corners':'false'},
                       {'conv_bias':1}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):micro(**kwargs)

    def test_spatial_empty_and_degenerate_norm_rejected(self):
        for shape in [(0,2,3),(-1,2,3),(1,2),(True,2,3)]:
            with self.assertRaises(ValueError):micro().validate_spatial(shape)
        c=micro(channels=(2,2,2),norm_groups=2)
        with self.assertRaisesRegex(ValueError,'per sample/group'):c.validate_spatial((1,1,1))
        self.assertEqual(micro().validate_spatial((1,1,1)),((1,1,1),)*3)

    def test_metadata_shape_audit_without_volume_or_tensor(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('shape_audit',ROOT/'scripts/audit_backbone_shapes.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rows=module.audit_shapes([{'case_id':'synthetic','candidate_id':'A','shape_dhw':[17,18,19]}],micro())
        self.assertEqual(rows[0]['encoder_dhw'],[[17,18,19],[9,9,10],[5,5,5]])
        self.assertEqual(rows[0]['decoder_dhw'],[[9,9,10],[17,18,19]])
        self.assertEqual(rows[0]['input_padding_voxels'],0)
        with self.assertRaises(ValueError):module.audit_shapes([],micro())


@unittest.skipUnless(TORCH_AVAILABLE,'isolated PyTorch CPU environment required')
class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):torch.manual_seed(1337)

    def test_example_interfaces_and_no_input_padding(self):
        config=micro();model=UNetBackbone3D(config)
        image=torch.randn(1,1,17,18,19);seen=[]
        handle=model.encoder.blocks[0].register_forward_pre_hook(lambda _,args:seen.append(tuple(args[0].shape)))
        f,skips=model.encoder(image);handle.remove()
        self.assertEqual(seen,[tuple(image.shape)])
        self.assertEqual(tuple(f.shape),(1,16,5,5,5))
        self.assertEqual([tuple(s.shape) for s in skips],[(1,4,17,18,19),(1,8,9,9,10)])
        self.assertEqual(tuple(model.decoder(f,skips).shape),(1,16,17,18,19))
        self.assertTrue(f.requires_grad and all(s.requires_grad for s in skips))

    def test_all_axis_parities_with_backward(self):
        model=UNetBackbone3D(micro())
        for shape in itertools.product((7,8),(9,10),(11,12)):
            with self.subTest(shape=shape):
                model.zero_grad(set_to_none=True)
                image=torch.randn(1,1,*shape,requires_grad=True)
                logits=model(image);self.assertEqual(tuple(logits.shape),(1,16,*shape))
                logits.square().mean().backward()
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(image.grad).all())
                self.assertGreater(float(image.grad.abs().sum()),0)

    def test_every_parameter_registered_and_receives_gradient(self):
        model=UNetBackbone3D(micro());x=torch.randn(2,1,9,10,11,requires_grad=True)
        model(x).square().mean().backward()
        inventory=parameter_inventory(model)
        # Independent hand count: encoder 13,612 + decoder/head 8,768.
        self.assertEqual(inventory['parameter_count'],22380)
        self.assertEqual(inventory['missing_gradients'],[])
        self.assertEqual(inventory['nonfinite_gradients'],[])
        self.assertEqual(inventory['zero_gradient_tensors'],[])
        registered={id(p) for p in model.parameters()}
        components={id(p) for module in (model.encoder,model.decoder) for p in module.parameters()}
        self.assertEqual(registered,components)
        self.assertEqual(inventory['parameter_bytes'],inventory['gradient_bytes'])

    def test_singletons_and_odd_anisotropic_input(self):
        model=UNetBackbone3D(micro())
        for shape in [(1,1,1),(1,7,9),(9,1,7),(5,7,1)]:
            with self.subTest(shape=shape):
                x=torch.randn(1,1,*shape,requires_grad=True);y=model(x)
                self.assertEqual(tuple(y.shape),(1,16,*shape));y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_different_depth_width_and_axis_strides(self):
        configs=[micro(channels=(4,8),downsample_strides=((2,2,2),)),
                 micro(channels=(2,4,8,16),downsample_strides=((1,2,2),(2,2,2),(2,1,2)))]
        for config in configs:
            with self.subTest(channels=config.channels):
                model=UNetBackbone3D(config);x=torch.randn(1,1,9,13,15,requires_grad=True)
                f,skips=model.encoder(x)
                self.assertEqual(tuple(f.shape[2:]),config.spatial_pyramid((9,13,15))[-1])
                y=model.decoder(f,skips);self.assertEqual(tuple(y.shape),(1,16,9,13,15))
                y.square().mean().backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_no_norm_and_conv_bias_configuration(self):
        model=UNetBackbone3D(micro(normalization='none',conv_bias=True))
        y=model(torch.randn(1,1,7,9,11));y.square().mean().backward()
        self.assertFalse(any(isinstance(m,torch.nn.GroupNorm) for m in model.modules()))
        self.assertEqual(parameter_inventory(model)['missing_gradients'],[])

    def test_replaced_F_is_used_and_features_not_modified(self):
        model=UNetBackbone3D(micro());image=torch.randn(1,1,7,8,9,requires_grad=True)
        f,skips=model.encoder(image);f.retain_grad()
        for s in skips:s.retain_grad()
        saved=[t.detach().clone() for t in (f,*skips)]
        delta=(torch.randn_like(f)*.1).requires_grad_()
        initial=model.decoder(f,skips);modified=model.decoder(f+delta,skips)
        self.assertGreater(float((initial-modified).abs().max().detach()),1e-6)
        modified.square().mean().backward()
        for original,copy in zip((f,*skips),saved):
            torch.testing.assert_close(original.detach(),copy,rtol=0,atol=0)
            self.assertIsNotNone(original.grad);self.assertGreater(float(original.grad.abs().sum()),0)
        self.assertGreater(float(delta.grad.abs().sum()),0)
        self.assertGreater(float(image.grad.abs().sum()),0)

    def test_logits_are_raw_and_no_label_argument(self):
        model=UNetBackbone3D(micro());x=torch.randn(1,1,5,6,7)
        with torch.no_grad():model.decoder.logits.weight.zero_();model.decoder.logits.bias.fill_(-2)
        torch.testing.assert_close(model(x),torch.full((1,16,5,6,7),-2.))
        with self.assertRaises(TypeError):model(x,torch.zeros(1,5,6,7))

    def test_input_contract_rejection(self):
        model=UNetBackbone3D(micro())
        for x in [torch.ones(1,5,6,7),torch.ones(1,2,5,6,7),torch.ones(1,1,5,6,7,dtype=torch.int64),torch.empty(1,1,0,6,7)]:
            with self.subTest(shape=x.shape),self.assertRaises(ValueError):model(x)

    def test_decoder_contract_rejection(self):
        encoder=Encoder3D(micro());decoder=Decoder3D(micro());f,skips=encoder(torch.randn(1,1,7,8,9))
        for bad_f,bad_skips in [(f,skips[:1]),(f,tuple(reversed(skips))),
                               (f[:,:,:,:1,:],skips),(f.double(),skips),
                               (f.expand(2,-1,-1,-1,-1),skips)]:
            with self.assertRaises(ValueError):decoder(bad_f,bad_skips)

    def test_cpu_float64_and_state_dict_roundtrip(self):
        model=UNetBackbone3D(micro()).double().eval();x=torch.randn(1,1,5,6,7,dtype=torch.float64)
        other=UNetBackbone3D(micro()).double().eval();other.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad():
            y=model(x);torch.testing.assert_close(y,other(x),atol=0,rtol=0)
        self.assertEqual(y.dtype,torch.float64);self.assertEqual(y.device.type,'cpu')

    def test_finite_difference_directional_input_gradient(self):
        model=UNetBackbone3D(micro(channels=(2,4),downsample_strides=((2,2,2),),normalization='none')).double()
        x=torch.randn(1,1,4,5,6,dtype=torch.float64,requires_grad=True)
        direction=torch.randn_like(x);direction/=direction.norm()
        loss=model(x).square().mean();grad=torch.autograd.grad(loss,x)[0]
        eps=1e-6
        with torch.no_grad():numeric=(model(x+eps*direction).square().mean()-model(x-eps*direction).square().mean())/(2*eps)
        analytic=(grad*direction).sum()
        torch.testing.assert_close(analytic,numeric,rtol=1e-3,atol=1e-8)

    def test_boundary_voxels_reach_deepest_features(self):
        config=micro(normalization='none');encoder=Encoder3D(config)
        with torch.no_grad():
            for p in encoder.parameters():p.fill_(.01)
            for shape in [(7,8,9),(8,9,10)]:
                baseline=encoder(torch.zeros(1,1,*shape)).deepest
                for corner in itertools.product(*[(0,n-1) for n in shape]):
                    image=torch.zeros(1,1,*shape);image[(0,0,*corner)]=1
                    self.assertGreater(float((encoder(image).deepest-baseline).abs().sum()),0)

    def test_interpolation_coordinate_ramp_all_axes(self):
        small=(3,4,5);large=(6,7,10)
        for aligned in (False,True):
            for axis in range(3):
                ramp=torch.arange(small[axis],dtype=torch.float64)
                axes=[1,1,1];axes[axis]=small[axis]
                source=ramp.reshape(*axes).expand(small)[None,None]
                result=resize_to_skip(source,large,align_corners=aligned)
                idx=torch.arange(large[axis],dtype=torch.float64)
                expected=idx*(small[axis]-1)/(large[axis]-1) if aligned else ((idx+.5)*small[axis]/large[axis]-.5).clamp(0,small[axis]-1)
                axes[axis]=large[axis]
                reference=expected.reshape(*axes).expand(large)[None,None]
                torch.testing.assert_close(result,reference,atol=1e-12,rtol=1e-12)

    def test_strided_center_kernel_phase_is_explicit(self):
        # A strided convolution samples nominal centers 0,s,2s...; size-based
        # interpolation has a separate phase. This test does NOT assert inversion.
        conv=torch.nn.Conv3d(1,1,3,stride=2,padding=1,bias=False).double()
        with torch.no_grad():conv.weight.zero_();conv.weight[0,0,1,1,1]=1
        for n in (7,8):
            ramp=torch.arange(n,dtype=torch.float64).reshape(1,1,n,1,1).expand(1,1,n,3,3)
            coarse=conv(ramp)
            torch.testing.assert_close(coarse[0,0,:,0,0],torch.arange(0,n,2,dtype=torch.float64))
            restored=resize_to_skip(coarse,(n,3,3),align_corners=False)
            expected=(((torch.arange(n,dtype=torch.float64)+.5)*math.ceil(n/2)/n-.5).clamp(0,math.ceil(n/2)-1))*2
            torch.testing.assert_close(restored[0,0,:,0,0],expected)
            self.assertGreater(float((restored-ramp).abs().max().detach()),0)

    def test_interpolation_gradient_and_singleton(self):
        source=torch.randn(1,1,2,3,2,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:resize_to_skip(x,(3,4,5),align_corners=False),(source,),fast_mode=True))
        one=torch.ones(1,1,1,1,1,requires_grad=True)
        output=resize_to_skip(one,(3,4,5),align_corners=False)
        torch.testing.assert_close(output,torch.ones_like(output));output.sum().backward()
        torch.testing.assert_close(one.grad,torch.full_like(one.grad,60),rtol=1e-6,atol=1e-6)

    def test_cpu_memory_interface_never_calls_cuda(self):
        with (patch.object(torch.cuda,'synchronize',side_effect=AssertionError('CPU called CUDA')),
              patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=AssertionError('CPU called CUDA'))):
            monitor=DeviceMemoryMonitor('cpu')
            with self.assertRaises(RuntimeError):monitor.snapshot('too_early')
            monitor.begin();data=monitor.snapshot('test')
        self.assertEqual(data['backend'],'cpu');self.assertIsNone(data['peak_allocated_bytes'])


if __name__=='__main__':unittest.main()
